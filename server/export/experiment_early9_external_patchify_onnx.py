#!/usr/bin/env python3
"""Early-9 P0: move lossless 16x16 patch packing OUTSIDE RKNN.

Must be installed at EviRoute_edge/export/experiment_early9_external_patchify_onnx.py.
Input changes from RGB NCHW [1,3,224,224] to packed NCHW [1,768,1,196].
Weights, prefix tokens, RoPE, blocks 0..8, mu/std and scorer remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from export.export_eviroute_early9_p0_onnx import (
    EviRouteEarly9P0Deploy, CHECKPOINT,
    OUT_DIR as BASE_ONNX_DIR, sha256_file, cosine, topk_logit,
)
from evaluation.eval_early9_p0 import load_model as load_p0_reference

OUT_DIR = ROOT / 'export' / 'early9_external_patchify_onnx'
ONNX_PATH = OUT_DIR / 'eviroute_dinov3b_early9_external_patchify_224.onnx'


def pack_rgb_to_nchw(x: np.ndarray) -> np.ndarray:
    """RGB NCHW FP32 -> contiguous packed patches NCHW for 1x1 conv.

    No normalization is applied; RGB preprocessing matches the original model.
    Patch index order: row-major (14x14), channel-major inside each 16x16 patch.
    """
    if x.shape != (1, 3, 224, 224) or x.dtype != np.float32:
        raise ValueError(f'Expected RGB float32 (1,3,224,224), got {x.shape}, {x.dtype}')
    packed = np.ascontiguousarray(
        x.reshape(1, 3, 14, 16, 14, 16)
         .transpose(0, 1, 3, 5, 2, 4)
         .reshape(1, 768, 1, 196)
    )
    if packed.shape != (1, 768, 1, 196):
        raise RuntimeError(f'Unexpected packed shape: {packed.shape}')
    return packed


class ExternalPatchProjection(nn.Module):
    """A 1x1 Conv projects externally packed [B,768,1,196] pixel patches.

    The 1x1 Conv uses exactly the original Conv2d patch-embedding weights.
    Returns [B,14,14,768] for the unchanged _prefix_embed() in golden code.
    """

    def __init__(self, original_patch_embed: nn.Module):
        super().__init__()
        original = original_patch_embed.proj
        if not isinstance(original, nn.Conv2d):
            raise TypeError(f'Expected nn.Conv2d: {type(original)}')
        if (original.in_channels, original.out_channels, original.kernel_size,
                original.stride, original.padding, original.groups) != (
                3, 768, (16, 16), (16, 16), (0, 0), 1):
            raise RuntimeError('Unexpected original patch embedding definition')
        if hasattr(original_patch_embed, 'norm') and not isinstance(original_patch_embed.norm, nn.Identity):
            raise RuntimeError('Non-identity patch embedding norm; cannot transfer directly')
        self.proj = nn.Conv2d(768, 768, kernel_size=1, bias=original.bias is not None)
        with torch.no_grad():
            self.proj.weight.copy_(original.weight.detach().reshape(768, 768, 1, 1))
            if original.bias is not None:
                self.proj.bias.copy_(original.bias.detach())
        self.eval().requires_grad_(False)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 4 or tuple(packed.shape[1:]) != (768, 1, 196):
            raise ValueError(f'Expected [B,768,1,196], got {tuple(packed.shape)}')
        b = packed.shape[0]
        y = self.proj(packed)
        return y.permute(0, 2, 3, 1).reshape(b, 14, 14, 768)


def report_stats(ref: np.ndarray, out: np.ndarray) -> dict:
    if ref.shape != (1, 196) or out.shape != (1, 196):
        raise ValueError((ref.shape, out.shape))
    if not np.isfinite(ref).all() or not np.isfinite(out).all():
        raise ValueError("Non-finite patch scores")
    delta = np.abs(ref.astype(np.float64) - out.astype(np.float64))
    ia = set(np.argsort(-ref[0])[:10].tolist())
    ib = set(np.argsort(-out[0])[:10].tolist())
    return {
        "cosine": cosine(ref, out),
        "mae": float(delta.mean()),
        "max_abs": float(delta.max()),
        "logit_error": abs(topk_logit(ref) - topk_logit(out)),
        "top10_overlap": len(ia & ib),
        "prediction_match": (topk_logit(ref) >= 0) == (topk_logit(out) >= 0),
    }


def aggregate(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "min_cosine": float(min(r["cosine"] for r in rows)),
        "mean_mae": float(np.mean([r["mae"] for r in rows])),
        "max_abs": float(max(r["max_abs"] for r in rows)),
        "max_logit_error": float(max(r["logit_error"] for r in rows)),
        "mean_top10_overlap": float(np.mean([r["top10_overlap"] for r in rows])),
        "prediction_matches": sum(bool(r["prediction_match"]) for r in rows),
    }


def strict_pass(report: dict, *, name: str):
    if (report["min_cosine"] < 0.99999 or
        report["max_abs"] > 0.01 or
        report["max_logit_error"] > 0.01 or
        report["mean_top10_overlap"] < 9.5 or
        report["prediction_matches"] != report["cases"]):
        raise RuntimeError(f"{name} failed strict numerical gate: {report}")




def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['verify', 'export'], default='verify')
    p.add_argument('--num-check', type=int, default=12)
    p.add_argument('--ort-check', type=int, default=4)
    args = p.parse_args()
    if not (4 <= args.num_check <= 12):
        p.error('--num-check must be in [4,12]')
    if not (2 <= args.ort_check <= args.num_check):
        p.error('--ort-check must be in [2,num-check]')
    if not CHECKPOINT.is_file() or not BASE_ONNX_DIR.is_dir():
        raise FileNotFoundError('Missing Early-9 P0 checkpoint or existing golden samples')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    print('EARLY9 EXTERNAL PATCHIFY | input=1x768x1x196 FP32 NCHW')
    print('Checkpoint SHA256:', sha256_file(CHECKPOINT))
    reference, _, checkpoint = load_p0_reference(torch.device('cpu'))
    model = EviRouteEarly9P0Deploy(reference, checkpoint).cpu().eval()
    original_patch = model.patch_embed
    external_projection = ExternalPatchProjection(original_patch).cpu().eval()

    rows_before, rows_after, rgb_images, packed_images, baseline_scores = [], [], [], [], []
    for k in range(args.num_check):
        in_path = BASE_ONNX_DIR / f'case_{k:02d}_input.npy'
        ref_path = BASE_ONNX_DIR / f'case_{k:02d}_pytorch_scores.npy'
        if not (in_path.is_file() and ref_path.is_file()):
            raise FileNotFoundError(f'Missing golden pair for case {k}: {in_path}, {ref_path}')
        img = np.load(in_path, allow_pickle=False).astype(np.float32, copy=False)
        ref = np.load(ref_path, allow_pickle=False).astype(np.float32, copy=False).reshape(1, 196)
        packed = pack_rgb_to_nchw(img)
        with torch.inference_mode():
            original = model(torch.from_numpy(img)).detach().numpy().astype(np.float32)
        rows_before.append(report_stats(ref, original))
        rgb_images.append(img)
        packed_images.append(packed)
        baseline_scores.append(original)
    before = aggregate(rows_before)
    strict_pass(before, name='Unmodified merged model vs golden PyTorch')
    print('Original merged vs golden:', json.dumps(before))

    # Replace ONLY the patch embedding input boundary and projection operator.
    model.patch_embed = external_projection
    # Local algebraic test against the original patch embedding, BEFORE the transformer.
    with torch.inference_mode():
        orig_tokens = original_patch(torch.from_numpy(rgb_images[0]))
        ext_tokens = external_projection(torch.from_numpy(packed_images[0]))
    patch_abs = (orig_tokens.float() - ext_tokens.float()).abs().max().item()
    print('Original patch embedding vs external 1x1 projection max_abs:', patch_abs)
    if not np.isfinite(patch_abs) or patch_abs > 0.001:
        raise RuntimeError('External projection patch-level equivalence failed')

    for k, (packed, original) in enumerate(zip(packed_images, baseline_scores)):
        with torch.inference_mode():
            candidate = model(torch.from_numpy(packed)).detach().numpy().astype(np.float32)
        stat = report_stats(original, candidate)
        rows_after.append(stat)
        np.save(OUT_DIR / f'case_{k:02d}_packed_input.npy', packed)
        np.save(OUT_DIR / f'case_{k:02d}_pytorch_scores.npy', original)
        np.save(OUT_DIR / f'case_{k:02d}_external_scores.npy', candidate)
        print(f'case={k:02d} cos={stat["cosine"]:.9f} max_abs={stat["max_abs"]:.6f} '
              f'logit_error={stat["logit_error"]:.6f} Top10={stat["top10_overlap"]}/10')
    merge_report = aggregate(rows_after)
    (OUT_DIR / 'merge_check.json').write_text(json.dumps(merge_report, indent=2) + '\n')
    print('EXTERNAL PATCHIFY EQUIVALENCE:', json.dumps(merge_report, indent=2))
    strict_pass(merge_report, name='External patchify vs unmodified merged model')
    print('EARLY9 EXTERNAL PATCHIFY PYTORCH: PASS')
    if args.phase == 'verify':
        print('VERIFY COMPLETE; continue with --phase export')
        return

    import onnx
    import onnxruntime as ort
    with torch.inference_mode():
        torch.onnx.export(
            model, torch.from_numpy(packed_images[0]), str(ONNX_PATH),
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['packed_patch_pixels'], output_names=['patch_scores'],
            dynamic_axes=None, dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
    print('ONNX MiB:', round(ONNX_PATH.stat().st_size / 1024**2, 3))
    session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    ort_input = session.get_inputs()
    ort_output = session.get_outputs()
    if (len(ort_input) != 1 or len(ort_output) != 1 or
            ort_input[0].shape != [1, 768, 1, 196]):
        raise RuntimeError(f'Unexpected ONNX IO: {ort_input}, {ort_output}')
    ort_rows = []
    for k in range(args.ort_check):
        y = session.run([ort_output[0].name], {ort_input[0].name: packed_images[k]})[0]
        stat = report_stats(baseline_scores[k], np.asarray(y, dtype=np.float32).reshape(1, 196))
        ort_rows.append(stat)
        print(f'ORT case={k:02d} cos={stat["cosine"]:.9f} max_abs={stat["max_abs"]:.6f} '
              f'logit_error={stat["logit_error"]:.6f} Top10={stat["top10_overlap"]}/10')
    ort_report = aggregate(ort_rows)
    (OUT_DIR / 'onnx_check.json').write_text(json.dumps(ort_report, indent=2) + '\n')
    strict_pass(ort_report, name='External patchify ONNXRuntime vs unmodified merged model')
    print('EARLY9 EXTERNAL PATCHIFY ONNX: PASS')
    metadata = {
        'experiment': 'early9_external_patchify_v2',
        'checkpoint_sha256': sha256_file(CHECKPOINT),
        'onnx_sha256': sha256_file(ONNX_PATH),
        'blocks': 9, 'lora_blocks': [6, 7, 8],
        'input_shape': [1, 768, 1, 196], 'input_dtype': 'float32',
        'input_format': 'nchw',
        'external_preprocess': 'np.ascontiguousarray(x.reshape(1,3,14,16,14,16).transpose(0,1,3,5,2,4).reshape(1,768,1,196))',
        'output_shape': [1, 196],
        'topk': 10,
    }
    (OUT_DIR / 'export_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('EARLY9 EXTERNAL PATCHIFY EXPORT: PASS')


if __name__ == '__main__':
    main()
