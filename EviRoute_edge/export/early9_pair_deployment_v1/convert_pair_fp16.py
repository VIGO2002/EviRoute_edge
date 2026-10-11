#!/usr/bin/env python3
"""Convert one verified paired Early-9 External Patchify ONNX to RK3588 FP16.

Run only inside AutoDL's rknn232 environment (RKNN Toolkit2 2.3.2).
The same operator/input/numerical gates as the successful P0 converter apply.
"""
from __future__ import annotations
import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from export.early9_pair_deployment_v1.pair_common import (
    OUTPUT_BASE,CHECKPOINTS,sha256,stats,save_json,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=list(CHECKPOINTS),required=True)
    p.add_argument('--sim-cases',type=int,default=8)
    p.add_argument('--build-only',action='store_true',help='Export RKNN without simulator; NOT considered validated')
    a=p.parse_args()
    if a.sim_cases<2 or a.sim_cases>16 or a.sim_cases%2:
        p.error('--sim-cases must be even and in [2,16]')
    base=ROOT/OUTPUT_BASE/a.arm
    metadata_path=base/'export_metadata.json'
    model_path=base/'early9_external_patchify_224.onnx'
    ckpt_path=ROOT/CHECKPOINTS[a.arm][0]
    if not (metadata_path.is_file() and model_path.is_file() and ckpt_path.is_file()):
        raise FileNotFoundError('Export ONNX+metadata+checkpoint first')
    meta=json.loads(metadata_path.read_text())
    if (meta['status']!='verified_onnx' or meta['experiment']!='early9_matched_pair_external_patchify_v1'
            or meta['arm']!=a.arm or meta['input_shape'] != [1,768,1,196]
            or meta['output_shape'] != [1,196]):
        raise RuntimeError('Model metadata mismatch')
    if meta['checkpoint_sha256']!=sha256(ckpt_path) or sha256(ckpt_path)!=CHECKPOINTS[a.arm][1]:
        raise RuntimeError('Checkpoint identity mismatch')
    if sha256(model_path)!=meta['onnx_sha256']:
        raise RuntimeError('ONNX hash mismatch')
    # Fail closed if any numerical baseline has changed.
    for k in ('merge','onnxruntime'):
        r=meta['numerical_gates'][k]
        if r['min_cosine']<.99999 or r['max_abs']>.01 or r['max_logit_error']>.01:
            raise RuntimeError(f'{k} numerical gate invalid: {r}')
    print('VERIFIED ONNX INPUT:',str(model_path),'SHA256:',meta['onnx_sha256'],flush=True)
    from rknn.api import RKNN
    rknn=RKNN(verbose=True)
    out=base/'rk3588_fp16'
    out.mkdir(exist_ok=True)
    rknn_path=out/f'early9_{a.arm.lower()}_external_patchify_fp16.rknn'
    try:
        t0=time.perf_counter()
        ret=rknn.config(target_platform='rk3588')
        if ret!=0:raise RuntimeError(f'RKNN config failed: {ret}')
        ret=rknn.load_onnx(model=str(model_path))
        if ret!=0:raise RuntimeError(f'RKNN load_onnx failed: {ret}')
        ret=rknn.build(do_quantization=False)
        if ret!=0:raise RuntimeError(f'RKNN FP16 build failed: {ret}')
        ret=rknn.export_rknn(str(rknn_path))
        if ret!=0 or not rknn_path.is_file():raise RuntimeError(f'RKNN export failed: {ret}')
        print('RKNN FP16 BUILD: PASS | MiB:',round(rknn_path.stat().st_size/1024**2,2),flush=True)
        records=[]
        if not a.build_only:
            ret=rknn.init_runtime()
            if ret!=0:raise RuntimeError(f'RKNN simulator init failed: {ret}')
            half=a.sim_cases//2
            indices=list(range(half))+list(range(8,8+half))
            for k in indices:
                x=np.load(base/f'case_{k:02d}_packed_input.npy',allow_pickle=False)
                ref=np.load(base/'board_cases'/f'case_{k:02d}_pytorch_scores.npy',allow_pickle=False)
                if x.shape!=(1,768,1,196) or x.dtype!=np.float32 or ref.shape!=(1,196):
                    raise RuntimeError(f'Bad simulator case {k}')
                output=rknn.inference(inputs=[x],data_format=['nchw'])
                if output is None or len(output)!=1 or output[0] is None:
                    raise RuntimeError(f'Simulator case {k} failed')
                r=stats(ref,output[0]);r['case']=k
                records.append(r)
                print('SIM',k,r,flush=True)
                np.save(out/f'case_{k:02d}_sim_scores.npy',np.asarray(output[0],dtype=np.float32))
            pass_sim=all(r['cosine']>=.995 and r['top10_overlap']>=8 and r['prediction_match'] for r in records)
            if not pass_sim:
                save_json(out/'conversion_validation.json',{
                    'passed':False,'simulator_cases':records,
                    'note':'Numerical gate failed; DO NOT deploy until investigated'})
                raise RuntimeError('RKNN simulator numerical gate failed')
        else:
            pass_sim=False
        result={
            'status':'simulator_pass' if pass_sim else 'build_only_unverified',
            'passed':pass_sim,
            'arm':a.arm,'source_checkpoint_sha256':meta['checkpoint_sha256'],
            'source_onnx_sha256':meta['onnx_sha256'],
            'rk3588_model_sha256':sha256(rknn_path),
            'rknn_model_path':str(rknn_path),
            'seconds':round(time.perf_counter()-t0,3),
            'toolkit_note':'RKNN Toolkit2 in AutoDL rknn232; target rk3588; do_quantization=False',
            'simulator_cases':records,
        }
        save_json(out/'conversion_validation.json',result)
        print(('RKNN SIMULATOR: PASS' if pass_sim else 'RKNN BUILD ONLY; NOT VALIDATED'),flush=True)
        print('BOARD SHA256:',result['rk3588_model_sha256'],flush=True)
    finally:
        rknn.release()

if __name__=='__main__':main()
