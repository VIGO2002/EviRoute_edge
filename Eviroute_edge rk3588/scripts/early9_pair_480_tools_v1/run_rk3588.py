#!/usr/bin/env python3
"""RK3588: numerical audit over the frozen 480 images, using actual 3-core NPU inference."""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
from common import ARMS, DOMAINS, MODEL_SHA256, read_meta_inputs, sha256, verify_sha, json_save, top10, rgb_uint8_to_packed, region_counts

CORE_MAP={'auto':'NPU_CORE_AUTO','0':'NPU_CORE_0','1':'NPU_CORE_1','2':'NPU_CORE_2','0_1_2':'NPU_CORE_0_1_2'}


def check_transport(data_dir):
    root=Path(data_dir)
    checks=json.loads((root/'transport_sha256.json').read_text(encoding='utf-8'))
    for name in ('inputs_224_uint8.npz','references.npz','manifest.json'):
        verify_sha(root/name,checks[name],f'board data {name}')
    return checks


def compare_arrays(reference, predicted):
    a=np.asarray(reference,dtype=np.float64).reshape(-1)
    b=np.asarray(predicted,dtype=np.float64).reshape(-1)
    if a.size!=196 or b.size!=196 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('Bad patch scores')
    za,ia=top10(a);zb,ib=top10(b)
    den=np.linalg.norm(a)*np.linalg.norm(b)
    cosine=float(np.dot(a,b)/den) if den>0 else float('nan')
    return dict(cosine=cosine,mae=float(np.mean(np.abs(a-b))),
                max_abs=float(np.max(np.abs(a-b))),top10_overlap=int(len(set(ia)&set(ib))),
                py_logit=za,npu_logit=zb,logit_abs_error=abs(za-zb),
                prediction_match=bool((za>=0)==(zb>=0)),
                ref_b1=region_counts(ia)[0],board_b1=region_counts(ib)[0])


def infer(lite, rgb):
    packed=rgb_uint8_to_packed(rgb)
    output=lite.inference(inputs=[packed],data_format=['nchw'])
    if not isinstance(output,(list,tuple)) or len(output)!=1 or output[0] is None:
        raise RuntimeError('RKNN returned missing output')
    a=np.asarray(output[0],dtype=np.float32).reshape(-1)
    if a.size!=196 or not np.isfinite(a).all():
        raise RuntimeError('RKNN returned wrong/nonfinite patch scores')
    return a


def select_indices(labels,domains,limit):
    if limit==480:return np.arange(480,dtype=np.int64)
    if limit!=16:raise ValueError('Expected 16 or 480')
    output=[]
    for dom in DOMAINS:
        for label in (0,1):
            match=np.flatnonzero((domains==dom)&(labels==label))
            if match.size!=30:raise RuntimeError('Non-balanced source smoke selection')
            output.append(int(match[0]))
    return np.asarray(output,dtype=np.int64)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('evaluation/early9_pair_480_v1'))
    p.add_argument('--out-dir',type=Path,default=Path('results/early9_pair_480_v1'))
    p.add_argument('--core',choices=list(CORE_MAP),default='0_1_2')
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--limit',type=int,default=480,help='Only 16 for smoke; 480 for final report')
    args=p.parse_args()
    if platform.machine().lower() not in ('aarch64','arm64'):
        raise RuntimeError('Requires actual RK3588 aarch64, not simulator')
    if not 1<=args.limit<=480 or args.warmup<0:
        p.error('--limit must be 1..480; --warmup >=0')
    if args.limit!=480 and args.limit!=16:
        p.error('Only --limit 16 smoke or --limit 480 complete protocol supported')
    checks=check_transport(args.data_dir)
    meta,z,inputs=read_meta_inputs(args.data_dir)
    model_sha=verify_sha(args.model,MODEL_SHA256[args.arm],args.arm+' RKNN model')
    if meta['provenance']['rk3588_models'][args.arm]!=model_sha:
        raise RuntimeError('Model SHA256 mismatches sample provenance')
    # IMPORTANT: cache decompressed NPZ arrays ONCE; indexing an NpzFile
    # repeatedly would otherwise re-decompress the entire 480-image archive.
    rgb_images=inputs['rgb'].copy()
    ref_scores=z[args.arm+'_scores'].copy()
    selected_idx=select_indices(z['labels'],z['domain'],args.limit)
    sample_ids=z['global_index'][selected_idx].copy()
    out_dir=args.out_dir.resolve();out_dir.mkdir(parents=True,exist_ok=True)
    suffix='smoke16' if args.limit==16 else 'full480'
    outnpz=out_dir/f'{args.arm}_{suffix}_board_outputs.npz'
    outjson=out_dir/f'{args.arm}_{suffix}_board_report.json'
    if outnpz.exists() or outjson.exists():
        raise FileExistsError('Refusing to overwrite existing numerical report. Select a new --out-dir.')
    from rknnlite.api import RKNNLite
    if not hasattr(RKNNLite,CORE_MAP[args.core]):
        raise RuntimeError('RKNNLite does not support core '+args.core)
    lite=RKNNLite(verbose=False)
    results=[];durations=[]
    try:
        code=lite.load_rknn(str(args.model))
        if code!=0:raise RuntimeError(f'load_rknn failed ret={code}')
        code=lite.init_runtime(core_mask=getattr(RKNNLite,CORE_MAP[args.core]))
        if code!=0:raise RuntimeError(f'init_runtime failed ret={code}')
        print('RK3588 NPU READY:',args.arm,'core',args.core,'model sha256',model_sha,flush=True)
        for i in range(args.warmup):
            infer(lite,rgb_images[selected_idx[i%args.limit]])
        scores=np.empty((args.limit,196),dtype=np.float32)
        gid=sample_ids
        for i in range(args.limit):
            start=time.perf_counter()
            scores[i]=infer(lite,rgb_images[selected_idx[i]])
            ms=(time.perf_counter()-start)*1000
            durations.append(ms)
            m=compare_arrays(ref_scores[selected_idx[i]],scores[i])
            results.append(m)
            if (i+1)%30==0 or i+1==args.limit:
                print(f'{args.arm}: {i+1}/{args.limit} | avg_top10={np.mean([v["top10_overlap"] for v in results]):.4f}/10 '
                      f'pred_agree={sum(v["prediction_match"] for v in results)}/{i+1} '
                      f'latest_ms={ms:.2f}',flush=True)
        arr={'global_index':gid,'patch_scores':scores,'model_sha256':np.asarray(model_sha),
             'ref_input_sha256':np.asarray(checks['inputs_224_uint8.npz']),
             'ref_scores_sha256':np.asarray(checks['references.npz'])}
        np.savez_compressed(outnpz,**arr)
        summary={
            'status':'completed' if args.limit==480 else 'smoke_only', 'arm':args.arm,
            'n':args.limit,'model_sha256':model_sha,'core':args.core,
            'reference_npz_sha256':checks['references.npz'],
            'input_npz_sha256':checks['inputs_224_uint8.npz'],
            'board_outputs_sha256':sha256(outnpz),
            'min_cosine':float(min(m['cosine'] for m in results)),
            'mean_cosine':float(np.mean([m['cosine'] for m in results])),
            'mean_mae':float(np.mean([m['mae'] for m in results])),
            'max_abs_error':float(max(m['max_abs'] for m in results)),
            'max_logit_error':float(max(m['logit_abs_error'] for m in results)),
            'mean_top10_overlap':float(np.mean([m['top10_overlap'] for m in results])),
            'prediction_agreement':int(sum(m['prediction_match'] for m in results)),
            'inference_ms_mean':float(np.mean(durations)),
            'inference_ms_p95':float(np.percentile(durations,95)),
            'measurement':'pre-resized RGB uint8 preloaded; array reconstruction+CPU patchify+NPU inference; excludes file decode/disk/network/UI; different from sustained benchmark',
        }
        json_save(outjson,summary)
        print('\nBOARD EVAL480:',summary['status'].upper(),args.arm,flush=True)
        print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
        print('SAVED:',outnpz,outjson,flush=True)
    finally:
        lite.release()

if __name__=='__main__':main()
