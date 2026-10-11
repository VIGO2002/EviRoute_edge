#!/usr/bin/env python3
"""Export frozen R-BCE / R-Debias LoRA Early-9 using original verified P0 deployment modules.

This adds isolated outputs ONLY under export/early9_pair_deployment_v1/<arm>/.
The algorithm, LoRA merge, prefix/RoPE, scorer, feature normalization and
external patchify are reused from the existing validated P0 implementation.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'distillation'))
from export.early9_pair_deployment_v1.pair_common import (
    CHECKPOINTS, OUTPUT_BASE, pack_rgb_to_nchw, select_cases, validated_files,
    sha256, stats, summarize, strict_float_gate, frozen_top10_audit, save_json,
)


def build_models(arm):
    import torch
    import evaluation.eval_early9_p0 as ev
    from export.export_eviroute_early9_p0_onnx import EviRouteEarly9P0Deploy
    from export.experiment_early9_external_patchify_onnx import ExternalPatchProjection
    path=ROOT/CHECKPOINTS[arm][0]
    original=ev.P0_CKPT
    try:
        # The legacy loader is known-good but originally hardcodes its P0 path.
        # Its P0_CKPT is restored afterwards: never modify its source file.
        ev.P0_CKPT=path
        reference, image_set, checkpoint=ev.load_model(torch.device('cpu'))
    finally:
        ev.P0_CKPT=original
    if checkpoint.get('feature_block')!=8:
        raise RuntimeError('Unexpected checkpoint feature block')
    if checkpoint.get('lora_config',{}).get('blocks') != [6,7,8]:
        raise RuntimeError('Unexpected checkpoint LoRA block configuration')
    if checkpoint.get('lora_config',{}).get('targets') != ['qkv','proj']:
        raise RuntimeError('Unexpected checkpoint LoRA targets')
    # Reuse the proven LoRA merge implementation, including independent
    # base-weight matching and exact scorer, mu and std comparisons.
    merged=EviRouteEarly9P0Deploy(reference,checkpoint).cpu().eval()
    projection=ExternalPatchProjection(merged.patch_embed).cpu().eval()
    reference.eval()
    return reference, merged, projection, image_set


def input_image(image_set, entry):
    import torch
    ds=image_set([entry['path']],[entry['label']])
    im,_=ds[0]
    if tuple(im.shape)!=(3,224,224) or im.dtype!=torch.float32:
        raise RuntimeError('ImageSet preprocessing does not match original protocol')
    img=im.unsqueeze(0).contiguous().numpy()
    if not np.isfinite(img).all() or img.min()<0 or img.max()>1:
        raise RuntimeError('Invalid RGB value range')
    return np.ascontiguousarray(img)


def run(args):
    import torch
    files=validated_files(ROOT,args.arm)
    out=ROOT/OUTPUT_BASE/args.arm
    print('ARM:',args.arm,flush=True)
    print('SOURCE FILES / SHA256: PASS',flush=True)
    print(json.dumps({key: {'path':value,'sha256':sha256(value)} for key,value in files.items()},indent=2),flush=True)
    if args.phase=='preflight':
        print('PAIR DEPLOYMENT PREFLIGHT: PASS')
        return
    out.mkdir(parents=True,exist_ok=True)
    (out/'board_cases').mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    rows=select_cases(ROOT,args.arm,args.num_check)
    print('GOLDEN SOURCE DISTRIBUTION:', [(r['domain'],r['label']) for r in rows],flush=True)
    reference,merged,projection,image_set=build_models(args.arm)

    original_patch=merged.patch_embed
    before=[]; after=[]; golden_checks=[]; originals=[]; packed=[]; scores_ref=[]
    near_ties=[]
    for k,row in enumerate(rows):
        x=input_image(image_set,row)
        px=pack_rgb_to_nchw(x)
        with torch.inference_mode():
            z, scores, idx=reference(torch.from_numpy(x))
            current=scores.detach().cpu().numpy().reshape(1,196).astype(np.float32)
            ref_logit=float(z.detach().cpu().numpy().reshape(-1)[0])
            ref_idx=idx.detach().cpu().numpy().reshape(-1)
            check=frozen_top10_audit(ref_logit,ref_idx,current,
                                     row['frozen_logit'],row['frozen_top10'])
            if check['near_tie']:
                near_ties.append({'global_index':row['global_index'],**check})
            baseline=merged(torch.from_numpy(x)).detach().numpy().astype(np.float32).reshape(1,196)
        s=stats(current,baseline);before.append(s)
        originals.append(x);packed.append(px);scores_ref.append(current)
        golden_checks.append({'global_index':row['global_index'],**check})
        print(f'CHECK {k+1:02d}/{len(rows)} id={row["global_index"]} class={row["label"]} '
              f'ref_vs_merge={s["max_abs"]:.6g} cos={s["cosine"]:.9f} '
              f'top10={s["top10_overlap"]}/10',flush=True)
    if len(near_ties)>2:
        raise RuntimeError(f'Too many frozen Top10 boundary near-ties: {near_ties}')
    before_report=summarize(before)
    strict_float_gate(before_report,'independent LoRA reference vs merged backbone')
    print('PAIR LoRA MERGE AUDIT: PASS',json.dumps(before_report),flush=True)

    # Use existing, *exact* patch projection transform, not a newly approximated patchify.
    merged.patch_embed=projection
    with torch.inference_mode():
        orig_token=original_patch(torch.from_numpy(originals[0]))
        packed_token=projection(torch.from_numpy(packed[0]))
        p_error=float((orig_token.float()-packed_token.float()).abs().max().item())
    if not np.isfinite(p_error) or p_error > .001:
        raise RuntimeError(f'External patch projection mismatch: {p_error}')

    for k,(px,ref) in enumerate(zip(packed,scores_ref)):
        with torch.inference_mode():
            y=merged(torch.from_numpy(px)).detach().numpy().astype(np.float32).reshape(1,196)
        s=stats(ref,y)
        after.append(s)
        np.save(out/'board_cases'/f'case_{k:02d}_input.npy',originals[k])
        np.save(out/f'case_{k:02d}_packed_input.npy',px)
        np.save(out/'board_cases'/f'case_{k:02d}_pytorch_scores.npy',ref)
        print(f'EXTERNAL {k+1:02d}/{len(packed)} max_abs={s["max_abs"]:.6g} '
              f'logit_err={s["logit_error"]:.6g} top10={s["top10_overlap"]}/10',flush=True)
    merged_report=summarize(after)
    strict_float_gate(merged_report,'external patchify vs original independent LoRA')
    save_json(out/'merge_check.json',merged_report)
    save_json(out/'case_manifest.json',{'cases':rows, 'frozen_checks':golden_checks,
                                        'near_ties':near_ties,'patch_projection_max_abs':p_error,
                                        'checkpoint_sha256':sha256(files['checkpoint'])})
    print('PAIR EXTERNAL PATCHIFY PYTORCH: PASS',json.dumps(merged_report),flush=True)
    if args.phase=='verify':
        print('VERIFY COMPLETE: next run --phase export',flush=True)
        return

    import onnx
    import onnxruntime as ort
    model_path=out/'early9_external_patchify_224.onnx'
    print('EXPORTING ONNX:',model_path,flush=True)
    with torch.inference_mode():
        torch.onnx.export(merged,torch.from_numpy(packed[0]),str(model_path),
                          export_params=True,opset_version=14,do_constant_folding=True,
                          input_names=['packed_patch_pixels'],output_names=['patch_scores'],
                          dynamic_axes=None,dynamo=False)
    onnx.checker.check_model(onnx.load(str(model_path)))
    sess=ort.InferenceSession(str(model_path),providers=['CPUExecutionProvider'])
    ins=sess.get_inputs();outs=sess.get_outputs()
    if len(ins)!=1 or len(outs)!=1 or ins[0].shape!=[1,768,1,196]:
        raise RuntimeError(f'Unexpected ONNX I/O: {ins}, {outs}')
    ort_cases=list(range(args.ort_check//2)) + list(range(8,8+args.ort_check//2))
    ort_stats=[]
    for k in ort_cases:
        y=sess.run([outs[0].name],{ins[0].name:packed[k]})[0]
        s=stats(scores_ref[k],y)
        ort_stats.append(s)
        print(f'ORT case={k:02d} cos={s["cosine"]:.9f} max_abs={s["max_abs"]:.6g} '
              f'logit_err={s["logit_error"]:.6g} top10={s["top10_overlap"]}/10',flush=True)
    ort_report=summarize(ort_stats)
    strict_float_gate(ort_report,'ONNXRuntime vs independent PyTorch')
    save_json(out/'onnx_check.json',ort_report)
    meta={
        'status':'verified_onnx', 'experiment':'early9_matched_pair_external_patchify_v1',
        'arm':args.arm,'checkpoint_sha256':sha256(files['checkpoint']),
        'paired_npz_sha256':sha256(files['npz']),'manifest_sha256':sha256(files['manifest']),
        'original_loader_sha256':sha256(files['loader']),
        'onnx_sha256':sha256(model_path),
        'input_shape':[1,768,1,196],'input_dtype':'float32','input_layout':'NCHW',
        'input_preprocess':'RGB resize BICUBIC to 224x224, [0,1] float32, no ImageNet norm, then exact external patchify',
        'output_shape':[1,196],'output_dtype':'float32','output_semantics':'fake-support patch scores, row-major 14x14',
        'topk':10,'lora_blocks':[6,7,8],'blocks':9,
        'numerical_gates':{'merge':merged_report,'onnxruntime':ort_report},
        'num_reference_images':len(rows),
    }
    save_json(out/'export_metadata.json',meta)
    print('PAIR EXTERNAL PATCHIFY ONNX: PASS',json.dumps({'arm':args.arm,'sha256':meta['onnx_sha256'],'onnx_mib':round(model_path.stat().st_size/1024**2,2)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=list(CHECKPOINTS),required=True)
    p.add_argument('--phase',choices=['preflight','verify','export'],default='preflight')
    p.add_argument('--num-check',type=int,default=16)
    p.add_argument('--ort-check',type=int,default=8)
    args=p.parse_args()
    if args.num_check!=16:p.error('For v1, --num-check must remain 16')
    if args.ort_check<2 or args.ort_check>16 or args.ort_check%2:
        p.error('--ort-check must be even and in [2,16]')
    run(args)

if __name__=='__main__': main()
