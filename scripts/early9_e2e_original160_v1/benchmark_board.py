#!/usr/bin/env python3
"""Run the previously verified JPEG/PNG RKNN application in paired, alternating order.

Runs do not modify models. Outputs a fresh run folder per arm/mode/repetition.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

MODES=('none','comparison')
ARMS=('R_BCE','R_Debias')

def run(args):
    root=Path(args.project_root).expanduser().resolve()
    input_dir=root/'images/full_pipeline_original160_v1'
    app=root/'scripts/early9_full_pipeline_v1/run_images.py'
    if not app.is_file():raise FileNotFoundError(app)
    manifest=input_dir/'manifest.json'
    m=json.loads(manifest.read_text(encoding='utf-8'))
    if m.get('status')!='prepared' or not m.get('original_encoded_bytes_copied_unchanged') or m.get('n_images')!=160:
        raise RuntimeError('Original160 manifest missing or not the exact 160-image corpus')
    if not (input_dir/'labels.csv').is_file():raise FileNotFoundError('Labels missing')
    for r in m['images']:
        if not (input_dir/r['relative_path']).is_file():raise FileNotFoundError('Missing original '+r['relative_path'])
    out=root/'results/early9_full_pipeline_v1'/args.tag
    if out.exists():raise FileExistsError('Refusing to overwrite/reuse existing benchmark: '+str(out))
    out.mkdir(parents=True)
    (out/'manifest.json').write_text(json.dumps({'input':str(input_dir),'n_images':160,'rounds':args.rounds,
                                                   'modes':list(MODES),'order':'alternating within each mode and round',
                                                   'measurement':'full JPEG/PNG with original encoded files; two save modes'},indent=2)+'\n')
    for rep in range(1,args.rounds+1):
        for mode_index,mode in enumerate(MODES):
            order=ARMS if ((rep+mode_index)%2==1) else tuple(reversed(ARMS))
            for arm in order:
                key=f'{mode}_{arm}_run{rep}'
                target=out/key
                logfile=out/f'{key}.log'
                cmd=[sys.executable,'-u',str(app),'--arm',arm,'--input',str(input_dir),
                     '--out-dir',str(target),'--core','0_1_2','--warmup',str(args.warmup),
                     '--save-visual',mode,'--progress-every','40']
                print('RUN',key,flush=True)
                with logfile.open('w',encoding='utf-8') as f:
                    p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=False)
                if p.returncode!=0:
                    print('FAILED:',key,'Inspect',logfile,flush=True)
                    raise RuntimeError(f'{key} failed, exit {p.returncode}')
                summary=json.loads((target/'summary.json').read_text())
                if summary['n_images']!=160 or summary['protocol']['save_visual']!=mode:
                    raise RuntimeError('Board app output did not follow the test protocol: '+key)
                print(key,'FPS=',round(summary['full_pipeline_images_per_second'],3),
                      'mean=',round(summary['phase_ms']['full_image_ms']['mean'],2),
                      'p95=',round(summary['phase_ms']['full_image_ms']['p95'],2),flush=True)
    print('PAIR ORIGINAL160 BENCHMARK: COMPLETE',out,flush=True)
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',default='/root/EviRoute_RK3588')
    ap.add_argument('--rounds',type=int,default=2)
    ap.add_argument('--warmup',type=int,default=20)
    ap.add_argument('--tag',required=True,help='NEW folder tag, e.g. formal160_20261010_a')
    a=ap.parse_args()
    if not 1<=a.rounds<=10:ap.error('rounds must be 1..10')
    run(a)
if __name__=='__main__':main()
