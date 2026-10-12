#!/usr/bin/env python3
"""Alternating 2 models x 2 image output modes; independent processes, no simultaneous RKNN models."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ARMS = ('R_BCE','R_Debias')
MODES = ('none','comparison')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('/root/EviRoute_RK3588'))
    p.add_argument('--input',type=Path,default=Path('/root/EviRoute_RK3588/images/full_pipeline_original160_v1'))
    p.add_argument('--tag',required=True,help='Fresh directory identifier (no overwrite)')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--queue-depth',type=int,default=3)
    args=p.parse_args(argv)
    if not 1<=args.rounds<=10 or not 1<=args.queue_depth<=16:
        p.error('rounds 1..10, queue-depth 1..16')
    root=args.project_root.expanduser().resolve()
    src=root/'scripts/early9_async_pipeline_v1/run_async.py'
    input_dir=args.input.expanduser().resolve()
    assert src.is_file(),f'Missing {src}'
    assert input_dir.is_dir(),f'Missing {input_dir}'
    output=root/'results/early9_async_pipeline_v1'/args.tag
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    output.mkdir(parents=True)
    (output/'benchmark_manifest.json').write_text(json.dumps({
        'input':str(input_dir),'tag':args.tag,'rounds':args.rounds,'queue_depth':args.queue_depth,
        'modes':MODES,'arms':ARMS,
        'note':'Alternating arm and mode order between rounds; no parallel model residency'},indent=2)+'\n')
    for rep in range(1,args.rounds+1):
        # Reverse mode order on even repetitions to reduce frequency/order confounds.
        modes = MODES if rep%2==1 else tuple(reversed(MODES))
        for mi, mode in enumerate(modes):
            arms = ARMS if (rep+mi)%2==1 else tuple(reversed(ARMS))
            for arm in arms:
                name=f'{mode}_{arm}_run{rep}'
                cmd=[sys.executable,'-u',str(src),'--arm',arm,
                     '--input',str(input_dir),'--out-dir',str(output/name),
                     '--core','0_1_2','--warmup',str(args.warmup),'--queue-depth',str(args.queue_depth),
                     '--save-visual',mode,'--progress-every','40']
                print('START',name,flush=True)
                log=output/f'{name}.log'
                with log.open('w') as stream:
                    result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=False)
                if result.returncode:
                    raise RuntimeError(f'{name}: exit={result.returncode}, inspect {log}')
                summary=json.loads((output/name/'summary.json').read_text())
                n=summary['n_images']
                if n !=160:
                    raise RuntimeError(f'{name}: expected 160 images, observed {n}')
                print(f"{name}: {summary['full_pipeline_images_per_second']:.3f} FPS, "
                      f"mean latency {summary['phase_ms']['full_image_ms']['mean']:.2f} ms",flush=True)
    print('ASYNC ORIGINAL160 BENCHMARK: COMPLETE',output,flush=True)

if __name__=='__main__': main()
