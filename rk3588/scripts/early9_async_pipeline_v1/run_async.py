#!/usr/bin/env python3
"""Bounded 3-stage threaded RK3588 Early-9 JPEG/PNG pipeline.

Producer: encoded file read/hash, Pillow RGB bicubic 224, external CPU patchify.
Main: ONLY owner of RKNNLite; sequential 196-score NPU inference + Top-10.
Writer: OPTIONAL Pillow visualization/PNG + ordered JSONL output.

The previously validated model, preprocessing and decision functions are imported
from scripts/early9_full_pipeline_v1/run_images.py WITHOUT modification.
This is a pipeline throughput experiment, not a claim of faster single-image latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / 'early9_full_pipeline_v1'))
import run_images as old


def _safe_put(q, obj, stop):
    while not stop.is_set():
        try:
            q.put(obj, timeout=0.2)
            return True
        except queue.Full:
            continue
    return False


def _check_model_and_input(args):
    if not args.allow_other_arch and platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError('Requires real aarch64 RK3588, except CPU-only unit tests')
    model = Path(args.model) if args.model else old.model_path_for_arm(args.project_root, args.arm)
    model = model.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    digest = old.sha256_file(model)
    if digest != old.MODEL_SHA256[args.arm]:
        raise RuntimeError(f'Model SHA256 mismatch: {digest}, expected {old.MODEL_SHA256[args.arm]}')
    paths, img_root = old.collect_images(args.input, recursive=not args.non_recursive)
    if args.max_images:
        paths = paths[:args.max_images]
    out = Path(args.out_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing output directory: {out}')
    return model, digest, paths, img_root, out


def execute(args, lite_class=None):
    model, digest, paths, img_root, out = _check_model_and_input(args)
    if lite_class is None:
        from rknnlite.api import RKNNLite
        lite_class = RKNNLite
    mask_name = old.CORE_MASK[args.core]
    if not hasattr(lite_class, mask_name):
        raise RuntimeError(f'RKNNLite does not support {mask_name}')
    lite = lite_class(verbose=False)
    loaded = False
    try:
        code = lite.load_rknn(str(model))
        if code != 0:
            raise RuntimeError(f'load_rknn failed: {code}')
        code = lite.init_runtime(core_mask=getattr(lite_class, mask_name))
        if code != 0:
            raise RuntimeError(f'init_runtime failed: {code}')
        loaded = True
        if args.warmup:
            rgb, _, _ = old.read_rgb224(paths[0], respect_exif=args.respect_exif)
            packed = old.rgb_uint8_to_packed(rgb)
            for _ in range(args.warmup):
                old.infer_scores(lite, packed)
        out.mkdir(parents=True, exist_ok=True)
        if args.save_visual != 'none':
            (out / 'visuals').mkdir(exist_ok=True)
        workq, writeq = queue.Queue(maxsize=args.queue_depth), queue.Queue(maxsize=args.queue_depth)
        stop = threading.Event()
        produced = threading.Event()
        inferred = threading.Event()
        failures = []
        records = []
        def worker_error(where, exc):
            failures.append((where, exc))
            stop.set()
        def producer():
            try:
                for i, path in enumerate(paths):
                    if stop.is_set():
                        break
                    t0 = time.perf_counter()
                    rgb, size, sha = old.read_rgb224(path, respect_exif=args.respect_exif)
                    t1 = time.perf_counter()
                    packed = old.rgb_uint8_to_packed(rgb)
                    t2 = time.perf_counter()
                    item = {'index':i,'path':path,'rgb':rgb,'size':size,'sha':sha,
                            'packed':packed,'t0':t0,'t1':t1,'t2':t2}
                    if not _safe_put(workq, item, stop):
                        break
            except Exception as exc:
                worker_error('producer',exc)
            finally:
                produced.set()
        def writer():
            try:
                with (out / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
                    while True:
                        try:
                            item = writeq.get(timeout=0.2)
                        except queue.Empty:
                            if inferred.is_set() and writeq.empty():
                                break
                            continue
                        i = item['index']
                        path = item['path']
                        t4 = item['t4']
                        tw0 = time.perf_counter()
                        visual_file = None
                        if args.save_visual != 'none':
                            stem = re.sub(r'[^a-zA-Z0-9_-]+', '_',path.stem)[:42] or 'image'
                            image_id = hashlib.sha1(str(path.relative_to(img_root)).encode()).hexdigest()[:10]
                            visual_file = f'visuals/{i:04d}_{stem}_{image_id}.png'
                            vis = old.create_visual(item['rgb'],item['ids'],args.save_visual)
                            vis.save(out/visual_file,format='PNG')
                        tw1 = time.perf_counter()
                        scores = item['scores']
                        ids = item['ids']
                        z = item['logit']
                        row = {
                          'index':i,'relative_path':str(path.relative_to(img_root)),
                          'source_path':str(path),'input_sha256':item['sha'],
                          'original_size_wh':list(item['size']),
                          'prediction':'FAKE' if z>=0 else 'REAL','fake_label':int(z>=0),
                          'logit':z,'probability_fake':old.sigmoid(z),
                          'top10_patch_indices':[int(v) for v in ids],
                          'top10_patch_rc':[[int(v//14),int(v%14)] for v in ids],
                          'top10_patch_scores':[float(scores[v]) for v in ids],
                          'top10_region_counts':old.evidence_stats(ids),
                          'visual_file':visual_file,
                          'timings_ms':{
                            'decode_resize_ms':(item['t1']-item['t0'])*1000,
                            'cpu_patchify_ms':(item['t2']-item['t1'])*1000,
                            'rknn_inference_ms':(item['t3']-item['tinfer_start'])*1000,
                            'cpu_top10_ms':(item['t4']-item['t3'])*1000,
                            'visual_write_ms':(tw1-tw0)*1000,
                            'full_image_ms':(tw1-item['t0'])*1000,
                            'producer_queue_wait_ms':(item['tinfer_start']-item['t2'])*1000,
                            'writer_queue_wait_ms':(tw0-t4)*1000,
                          },
                        }
                        if args.store_all_scores:
                            row['all_196_patch_scores'] = scores.tolist()
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                        records.append(row)
                        if (i+1)%args.progress_every==0 or i+1==len(paths):
                            print(f'{args.arm} ASYNC {i+1}/{len(paths)} | logit={z:+.4f} {row["prediction"]}',flush=True)
            except Exception as exc:
                worker_error('writer',exc)
        start_wall = time.perf_counter()
        temp_before = old.get_thermal()
        pt = threading.Thread(target=producer,name='decode-producer',daemon=True)
        wt = threading.Thread(target=writer,name='png-writer',daemon=True)
        wt.start(); pt.start()
        try:
            while True:
                if failures:
                    break
                try:
                    item = workq.get(timeout=0.2)
                except queue.Empty:
                    if produced.is_set() and workq.empty():
                        break
                    continue
                item['tinfer_start'] = time.perf_counter()
                item['scores'] = old.infer_scores(lite, item['packed'])
                item['t3'] = time.perf_counter()
                item['logit'], item['ids'] = old.logit_and_top10(item['scores'])
                item['t4'] = time.perf_counter()
                # Release packed array reference before queueing PNG job.
                del item['packed']
                if not _safe_put(writeq, item, stop):
                    break
        except Exception as exc:
            worker_error('inference',exc)
        finally:
            inferred.set()
            if failures:
                stop.set()
            pt.join(timeout=10)
            wt.join(timeout=10)
        if pt.is_alive() or wt.is_alive():
            stop.set()
            raise RuntimeError('Pipeline worker thread failed to stop')
        if failures:
            where, exc = failures[0]
            raise RuntimeError(f'{where} pipeline failed: {exc}') from exc
        wall_s = time.perf_counter()-start_wall
        if len(records)!=len(paths) or [r['index'] for r in records]!=list(range(len(paths))):
            raise RuntimeError('Missing, duplicate, or reordered image records')
        keys=('decode_resize_ms','cpu_patchify_ms','rknn_inference_ms','cpu_top10_ms',
              'visual_write_ms','full_image_ms','producer_queue_wait_ms','writer_queue_wait_ms')
        summary={
          'status':'complete','arm':args.arm,'model':str(model),'model_sha256':digest,
          'core':args.core,'n_images':len(paths),'input':str(Path(args.input).resolve()),
          'image_sort':'sorted by relative path','warmup':args.warmup,'queue_depth':args.queue_depth,
          'pipeline':'producer (decode+patchify) / main-thread NPU / writer (PNG and JSONL), bounded queues',
          'protocol':{
            'decode':'Pillow RGB conversion, BICUBIC resize (224,224), EXIF transpose '+('ON' if args.respect_exif else 'OFF'),
            'normalization':'float32 0..1 divide by float32(255); NO ImageNet norm',
            'packed_input':'[1,768,1,196] float32, 16x16 external patchify',
            'output':'196 NPU fake-support patch scores; top10 mean; logit>=0 -> FAKE',
            'top10_visual_meaning':'Top-10 highest fake-support patches, NOT artifact ground truth',
            'save_visual':args.save_visual,
            'time_includes':'wall-clock pipeline of encoded read/decode, packed preprocess, NPU, top10, optional PNG/JSONL',
            'time_excludes':'model load/init, warmup, final summary JSON, UI/network',
            'latency_note':'per-image full_image_ms includes queue wait; phase_ms means NOT additive under overlap',
          },
          'wall_seconds':wall_s,'full_pipeline_images_per_second':len(paths)/wall_s,
          'phase_ms':{key:old.summarize_ms([r['timings_ms'][key] for r in records]) for key in keys},
          'temperature_c_before':temp_before,'temperature_c_after':old.get_thermal(),
          'process_vmrss_kib_after':old.vmrss_kib(),
          'prediction_counts':{v:sum(r['prediction']==v for r in records) for v in ('REAL','FAKE')},
          'output':str(out),
        }
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('ASYNC PIPELINE: COMPLETE',args.arm)
        print(json.dumps({'n_images':len(paths),'full_pipeline_images_per_second':summary['full_pipeline_images_per_second'],
                         'phase_ms':summary['phase_ms'],'prediction_counts':summary['prediction_counts']},indent=2))
        return summary
    finally:
        if loaded:
            try:
                lite.release()
            except Exception as exc:
                print('WARNING: release failed:',exc,file=sys.stderr)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=sorted(old.MODEL_SHA256),required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--project-root',type=Path,default=Path('/root/EviRoute_RK3588'))
    p.add_argument('--model',type=Path)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--core',choices=list(old.CORE_MASK),default='0_1_2')
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--max-images',type=int,default=0)
    p.add_argument('--non-recursive',action='store_true')
    p.add_argument('--save-visual',choices=('none','comparison','overlay'),default='comparison')
    p.add_argument('--store-all-scores',action='store_true')
    p.add_argument('--respect-exif',action='store_true')
    p.add_argument('--queue-depth',type=int,default=3)
    p.add_argument('--progress-every',type=int,default=40)
    p.add_argument('--allow-other-arch',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.max_images<0 or args.warmup<0 or not 1<=args.queue_depth<=16 or args.progress_every<1:
        p.error('max-images,warmup>=0; queue-depth 1..16; progress-every>=1')
    return args

if __name__=='__main__':
    execute(parse_args())
