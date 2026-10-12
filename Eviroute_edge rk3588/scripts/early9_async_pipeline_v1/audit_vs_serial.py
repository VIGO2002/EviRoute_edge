#!/usr/bin/env python3
"""Require identical encoded inputs, logits, labels and Top-10 vs verified sequential baseline."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ARMS=('R_BCE','R_Debias')
MODES=('none','comparison')

def read_rows(run):
    rows=[json.loads(line) for line in (run/'predictions.jsonl').read_text().splitlines() if line.strip()]
    if len(rows)!=160:raise RuntimeError(f'Expected 160 images under {run}, got {len(rows)}')
    return rows

def audit(serial_root,async_root,rounds=3,logit_tol=1e-5):
    records=[]
    for rep in range(1,rounds+1):
        for mode in MODES:
            for arm in ARMS:
                name=f'{mode}_{arm}_run{rep}'
                old=serial_root/name
                new=async_root/name
                s=read_rows(old); a=read_rows(new)
                n_ok=0; top10_exact=0; max_diff=0.0
                for x,y in zip(s,a):
                    if (x['relative_path'],x['input_sha256']) != (y['relative_path'],y['input_sha256']):
                        raise RuntimeError(f'{name}: input identity/order mismatch at {x["index"]}')
                    d=abs(float(x['logit'])-float(y['logit']))
                    max_diff=max(max_diff,d)
                    if d>logit_tol:
                        raise RuntimeError(f'{name}: logit delta {d} exceeds tolerance {logit_tol}: {x["relative_path"]}')
                    if x['prediction']!=y['prediction']:
                        raise RuntimeError(f'{name}: prediction mismatch {x["relative_path"]}')
                    n_ok+=1
                    if x['top10_patch_indices'] != y['top10_patch_indices']:
                        raise RuntimeError(f'{name}: ordered Top-10 mismatch {x["relative_path"]}')
                    top10_exact+=1
                ss=json.loads((old/'summary.json').read_text())
                aa=json.loads((new/'summary.json').read_text())
                s_fps=float(ss['full_pipeline_images_per_second']);a_fps=float(aa['full_pipeline_images_per_second'])
                records.append({'run':name,'arm':arm,'mode':mode,'rep':rep,
                                'n':n_ok,'ordered_top10_exact':top10_exact,
                                'max_logit_delta':max_diff,'serial_fps':s_fps,'async_fps':a_fps,
                                'speedup_x':a_fps/s_fps,'serial_wall_s':float(ss['wall_seconds']),
                                'async_wall_s':float(aa['wall_seconds'])})
                print(f'{name}: SAME {n_ok}/160 ordered Top-10/logit; '
                      f'{s_fps:.3f} -> {a_fps:.3f} FPS ({a_fps/s_fps:.3f}x)')
    grouped={}
    for arm in ARMS:
        for mode in MODES:
            matching=[r for r in records if r['arm']==arm and r['mode']==mode]
            total_n=sum(r['n'] for r in matching)
            base=total_n/sum(r['serial_wall_s'] for r in matching)
            new=total_n/sum(r['async_wall_s'] for r in matching)
            grouped[f'{arm}__{mode}']={'serial_fps':base,'async_fps':new,'speedup_x':new/base,
                                       'n_images_evaluated':total_n}
    out=async_root/'audit_vs_serial.json'
    out.write_text(json.dumps({'status':'passed','comparison':grouped,'runs':records},indent=2)+'\n')
    with (async_root/'audit_vs_serial.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print('ASYNC VS SERIAL AUDIT: PASS',out)
    for key,v in grouped.items():
        print(f"{key}: serial={v['serial_fps']:.3f}, async={v['async_fps']:.3f}, x{v['speedup_x']:.3f}")
    return records

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--serial-root',type=Path,required=True)
    ap.add_argument('--async-root',type=Path,required=True)
    ap.add_argument('--rounds',type=int,default=3)
    ap.add_argument('--logit-tol',type=float,default=1e-5)
    a=ap.parse_args(argv)
    audit(a.serial_root,a.async_root,a.rounds,a.logit_tol)

if __name__=='__main__':main()
