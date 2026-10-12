#!/usr/bin/env python3
"""Summarize real-board full-pipeline repeated benchmarks with input-identity audits."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def analyze(folder:Path):
    folder=folder.expanduser().resolve()
    if not folder.is_dir():raise FileNotFoundError(folder)
    entries=[]
    first_identity=None
    for p in sorted(folder.glob('*/summary.json')):
        sub=p.parent
        s=json.loads(p.read_text())
        pred=sub/'predictions.jsonl'
        rows=[json.loads(x) for x in pred.read_text(encoding='utf-8').splitlines() if x.strip()]
        if s.get('status')!='complete' or len(rows)!=s['n_images'] or s['n_images']!=160:
            raise RuntimeError('Incomplete full pipeline results: '+str(sub))
        identity=[(x['relative_path'],x['input_sha256']) for x in rows]
        if len(set(identity))!=160:
            raise RuntimeError('Missing/duplicated images in '+str(sub))
        if first_identity is None:first_identity=identity
        elif first_identity!=identity:
            raise RuntimeError('Input identity/order/SHA mismatch: '+str(sub))
        mode=s['protocol']['save_visual']
        arm=s['arm']
        if mode not in ('none','comparison') or arm not in ('R_BCE','R_Debias'):
            raise RuntimeError('Unknown run mode or model: '+str(sub))
        entry={'run':sub.name,'arm':arm,'mode':mode,'n':s['n_images'],
               'fps':s['full_pipeline_images_per_second'],
               'full_ms_mean':s['phase_ms']['full_image_ms']['mean'],
               'full_ms_p95':s['phase_ms']['full_image_ms']['p95'],
               'npu_ms_mean':s['phase_ms']['rknn_inference_ms']['mean'],
               'decode_ms_mean':s['phase_ms']['decode_resize_ms']['mean'],
               'patchify_ms_mean':s['phase_ms']['cpu_patchify_ms']['mean'],
               'top10_ms_mean':s['phase_ms']['cpu_top10_ms']['mean'],
               'visual_write_ms_mean':s['phase_ms']['visual_write_ms']['mean'],
               'vmrss_kib':s.get('process_vmrss_kib_after'),
               'temp_before':s.get('temperature_c_before'),
               'temp_after':s.get('temperature_c_after')}
        entries.append(entry)
    if not entries:raise FileNotFoundError('No completed run summaries found under '+str(folder))
    grouped=defaultdict(list)
    for e in entries:grouped[(e['arm'],e['mode'])].append(e)
    if set(grouped)!={(a,m) for a in ('R_BCE','R_Debias') for m in ('none','comparison')}:
        raise RuntimeError('Expected all 2 arms x 2 modes')
    for v in grouped.values():
        if len(v)!=len(entries)//4:raise RuntimeError('Uneven repetitions')
    out=[]
    for (arm,mode),values in sorted(grouped.items()):
        fps=np.array([x['fps'] for x in values])
        o={'arm':arm,'mode':mode,'runs':len(values),'n_images_per_run':160,
           'fps_mean':round(float(fps.mean()),5),
           'fps_min':round(float(fps.min()),5),'fps_max':round(float(fps.max()),5),
           'fps_sd':round(float(fps.std(ddof=1)),5) if len(fps)>1 else None,
           'full_ms_mean':round(float(np.mean([x['full_ms_mean'] for x in values])),3),
           'full_ms_p95_mean':round(float(np.mean([x['full_ms_p95'] for x in values])),3),
           'npu_ms_mean':round(float(np.mean([x['npu_ms_mean'] for x in values])),3),
           'decode_ms_mean':round(float(np.mean([x['decode_ms_mean'] for x in values])),3),
           'visual_write_ms_mean':round(float(np.mean([x['visual_write_ms_mean'] for x in values])),3)}
        out.append(o)
        print(f'{arm:8} {mode:10} {o["runs"]} runs | FPS {o["fps_mean"]:.3f} '
              f'(range {o["fps_min"]:.3f}..{o["fps_max"]:.3f}) | '
              f'NPU {o["npu_ms_mean"]:.2f} ms | visual/write {o["visual_write_ms_mean"]:.2f} ms')
    with (folder/'formal160_runs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in entries[0] if k not in ('temp_before','temp_after')]);w.writeheader()
        for e in entries:w.writerow({k:v for k,v in e.items() if k not in ('temp_before','temp_after')})
    with (folder/'formal160_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    (folder/'formal160_summary.json').write_text(json.dumps({'status':'completed',
        'n_unique_original_images':160,'run_input_identity_and_sha_match':True,
        'method':'sequential original JPEG/PNG input; paired models; with/without evidence PNG writes',
        'caution':'exploratory previously inspected cohort; not unseen benchmark; fps does not include UI/network/model init',
        'groups':out,'runs':entries},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('ORIGINAL160 SUMMARY: PASS')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True)
    a=p.parse_args();analyze(a.folder)
if __name__=='__main__':main()
