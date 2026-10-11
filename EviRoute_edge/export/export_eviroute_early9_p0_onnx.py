#!/usr/bin/env python3
"""Export the *trained* Early-9 P0 LoRA model to static ONNX.

Installation location: <project>/export/export_eviroute_early9_p0_onnx.py

This program reuses the previously verified Early-7 deployment forward,
with nine blocks, the exact Early-9 pretrained weights, LoRA merging,
and the P0 scorer and normalization statistics.

Outputs are 196 *patch scores*. Top-10/logit/sigmoid are CPU postprocessing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "distillation"))

from export.validate_early7_deployment import EviRouteEarly7Deploy
from models.eviroute import EviRouteScorer
import train_early9_lora as training
from evaluation.eval_early9_p0 import load_model as load_p0_reference

OUT_DIR = ROOT / "export" / "early9_p0_onnx"
ONNX_PATH = OUT_DIR / "eviroute_dinov3b_early9_p0_224.onnx"
CHECKPOINT = ROOT / "distillation/lora_runs/P0_Early9_20k_classic/best.pt"
VAL_CACHE = ROOT / "distillation/depth_sweep_features/vitb_early9/val.pt"
EXPECTED_LORA_BLOCKS = (6, 7, 8)
EXPECTED_LORA_TARGETS = ("qkv", "proj")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64).reshape(-1)
    bb = b.astype(np.float64).reshape(-1)
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / den) if den else float("nan")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def topk_logit(scores: np.ndarray) -> float:
    return float(np.sort(scores.reshape(-1))[-10:].mean(dtype=np.float64))


class EviRouteEarly9P0Deploy(EviRouteEarly7Deploy):
    """Reuses known-good prefix/RoPE and forward implementation from Early-7.

    In contrast to the parent, it uses blocks[0:9] and merges P0 LoRA.
    Importantly, we DO NOT call EviRouteEarly7Deploy.__init__.
    """

    def __init__(self, reference: nn.Module, checkpoint: dict):
        nn.Module.__init__(self)
        if training.LORA_BLOCKS != EXPECTED_LORA_BLOCKS:
            raise RuntimeError("Unexpected training LoRA blocks")
        if training.MODEL_NAME != "vit_base_patch16_dinov3.lvd1689m":
            raise RuntimeError(f"Unexpected pretrained tag: {training.MODEL_NAME}")
        if int(checkpoint.get("feature_block", -1)) != 8:
            raise RuntimeError("P0 checkpoint feature block is not 8")
        if checkpoint.get("backbone") != training.MODEL_NAME:
            raise RuntimeError("P0 checkpoint backbone tag mismatch")

        config = checkpoint["lora_config"]
        if list(config["blocks"]) != list(EXPECTED_LORA_BLOCKS):
            raise RuntimeError("P0 checkpoint LoRA block mismatch")
        if list(config["targets"]) != list(EXPECTED_LORA_TARGETS):
            raise RuntimeError("P0 checkpoint LoRA target mismatch")

        import timm
        backbone = timm.create_model(
            training.MODEL_NAME,
            pretrained=True,
            img_size=224,
        )
        backbone.eval()
        if len(backbone.blocks) != 12 or int(backbone.num_features) != 768:
            raise RuntimeError("Backbone depth or width mismatch")

        self.num_prefix_tokens = int(getattr(backbone, "num_prefix_tokens", -1))
        if self.num_prefix_tokens != 5:
            raise RuntimeError("Expected CLS and 4 register tokens")

        # Names match EviRouteEarly7Deploy._prefix_embed / forward_backbone.
        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.reg_token = backbone.reg_token
        self.pos_embed = backbone.pos_embed
        self.pos_drop = backbone.pos_drop
        self.patch_drop = getattr(backbone, "patch_drop", None)
        self.norm_pre = getattr(backbone, "norm_pre", nn.Identity())
        self.blocks = nn.ModuleList(list(backbone.blocks[:9]))

        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
            patches = backbone.patch_embed(dummy)
            native_pos, rope = backbone._pos_embed(patches)
            manual_pos = self._prefix_embed(patches)
            prefix_error = (native_pos - manual_pos).abs().max().item()
            print(f"Prefix embedding max error: {prefix_error:.3e}")
            if prefix_error > 1e-7:
                raise RuntimeError("Manual prefix embedding differs from timm")
        self.register_buffer("rope_const", rope.detach().clone(), persistent=True)
        print("Frozen RoPE shape:", tuple(self.rope_const.shape))

        # Compare the independently instantiated pretrained weights to those
        # that were actually used by the verified P0 evaluation model.
        max_base_error = 0.0
        max_merge_error = 0.0
        total_delta_norm = 0.0
        with torch.no_grad():
            for block_idx in EXPECTED_LORA_BLOCKS:
                for target in EXPECTED_LORA_TARGETS:
                    wrapped = getattr(reference.backbone.blocks[block_idx].attn, target)
                    linear = getattr(self.blocks[block_idx].attn, target)
                    if not isinstance(linear, nn.Linear):
                        raise RuntimeError(f"Expected nn.Linear at {block_idx}.{target}")
                    if not isinstance(wrapped, training.LoRALinear):
                        raise RuntimeError(f"Missing LoRA wrapper at {block_idx}.{target}")
                    base_diff = (linear.weight - wrapped.base.weight).abs().max().item()
                    max_base_error = max(max_base_error, base_diff)
                    if base_diff > 1e-7:
                        raise RuntimeError(
                            f"Base pretrained weights do not match at {block_idx}.{target}: {base_diff}"
                        )
                    if (linear.bias is None) != (wrapped.base.bias is None):
                        raise RuntimeError("Bias layout mismatch")
                    if linear.bias is not None:
                        if not torch.equal(linear.bias, wrapped.base.bias):
                            raise RuntimeError("Base bias mismatch")
                    delta = (wrapped.B.detach() @ wrapped.A.detach()) * wrapped.scale
                    if not torch.isfinite(delta).all():
                        raise RuntimeError("Non-finite LoRA delta")
                    total_delta_norm += float(torch.linalg.vector_norm(delta).item())
                    linear.weight.add_(delta)
                    merged_err = (linear.weight - wrapped.weight.detach()).abs().max().item()
                    max_merge_error = max(max_merge_error, merged_err)
                    if merged_err > 1e-6:
                        raise RuntimeError("LoRA merge mismatch")
                    print(
                        f"Merged block={block_idx} {target}: "
                        f"delta_L2={torch.linalg.vector_norm(delta).item():.6f}; "
                        f"base_max_err={base_diff:.3e}; merge_max_err={merged_err:.3e}"
                    )
        if total_delta_norm <= 0:
            raise RuntimeError("No nonzero LoRA weights merged")
        print("LoRA merge: PASS", "total_delta_L2_sum=", round(total_delta_norm, 6))
        print("Max pretrained base difference:", max_base_error)
        print("Max merged weight difference:", max_merge_error)

        self.scorer = EviRouteScorer(feat_dim=768, hidden_dim=256, dropout=0.1)
        self.scorer.load_state_dict(checkpoint["head_state_dict"], strict=True)
        self.register_buffer("mu", checkpoint["mu"].float().view(1, 1, 768))
        self.register_buffer("std", checkpoint["std"].float().view(1, 1, 768))
        if not bool(torch.isfinite(self.mu).all() and torch.isfinite(self.std).all()):
            raise RuntimeError("Nonfinite feature normalization")
        if bool((self.std < 1e-6).any()):
            raise RuntimeError("Invalid feature std")
        if not torch.equal(self.mu.reshape(-1), reference.mu.cpu().reshape(-1)):
            raise RuntimeError("P0 feature mu mismatch")
        if not torch.equal(self.std.reshape(-1), reference.std.cpu().reshape(-1)):
            raise RuntimeError("P0 feature std mismatch")
        # Check trained head, not the frozen baseline scorer.
        for key, value in self.scorer.state_dict().items():
            original = reference.head.state_dict()[key]
            if not torch.equal(value, original):
                raise RuntimeError(f"P0 head weight mismatch: {key}")

        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)
        # torch.onnx.export will trace forward() inherited from Early-7.
        # That forward emits ONLY 196 trained P0 patch scores.
        print("Deployment model blocks:", len(self.blocks))
        print("Scorer and mu/std: P0 checkpoint verified")


def chosen_validation_samples(n: int):
    cache = torch.load(VAL_CACHE, map_location="cpu", weights_only=False, mmap=True)
    paths = [str(p) for p in cache["paths"]]
    labels = np.asarray(cache["labels"], dtype=np.int64).reshape(-1)
    if len(paths) != len(labels) or len(paths) != 600:
        raise RuntimeError("Unexpected Early-9 validation cache size")
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError("Nonbinary validation labels")
    if n < 4 or n > len(labels) or n % 2:
        raise ValueError("--num-check must be an even integer in [4, 600]")
    half = n // 2
    picks = []
    for label in (0, 1):
        available = np.flatnonzero(labels == label)
        if len(available) < half:
            raise RuntimeError("Not enough images per label")
        chosen = available[np.linspace(0, len(available)-1, half, dtype=int)]
        picks.extend(chosen.tolist())
    return paths, labels, picks


def compare_pytorch_models(reference, deployment, image_set, paths, labels, indices):
    """Compare independent LoRA P0 model vs merged static deployment."""
    from torch.utils.data import DataLoader, Subset
    dataset = image_set(paths, labels)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cos_list, max_list, mae_list, logits, overlap_list = [], [], [], [], []
    with torch.inference_mode():
        for k, ((x, _), index) in enumerate(zip(loader, indices)):
            pt_logit, pt_scores, pt_top10 = reference(x)
            y = deployment(x)
            if tuple(y.shape) != (1, 196):
                raise RuntimeError("Unexpected merged model output shape")
            a = pt_scores.numpy().astype(np.float32)
            b = y.numpy().astype(np.float32)
            diff = np.abs(a.astype(np.float64)-b.astype(np.float64))
            sim = cosine(a, b)
            a_top = set(np.argsort(-a[0])[:10])
            b_top = set(np.argsort(-b[0])[:10])
            ovlp = len(a_top & b_top)
            logit_err = abs(float(pt_logit.item()) - topk_logit(b))
            cos_list.append(sim)
            max_list.append(float(diff.max()))
            mae_list.append(float(diff.mean()))
            logits.append(logit_err)
            overlap_list.append(ovlp)
            np.save(OUT_DIR / f"case_{k:02d}_input.npy", x.numpy().astype(np.float32))
            np.save(OUT_DIR / f"case_{k:02d}_pytorch_scores.npy", a)
            print(
                f"Check {k:02d} label={labels[index]} cosine={sim:.9f} "
                f"max_abs={diff.max():.6f} logit_err={logit_err:.6f} Top10={ovlp}/10"
            )
    report = dict(
        samples=len(indices), min_cosine=float(min(cos_list)),
        max_abs=float(max(max_list)), mean_mae=float(np.mean(mae_list)),
        max_logit_error=float(max(logits)), mean_top10_overlap=float(np.mean(overlap_list)),
    )
    (OUT_DIR / "merge_check.json").write_text(json.dumps(report, indent=2)+"\n")
    print("\nLoRA -> merged deployment numerical audit:", json.dumps(report, indent=2))
    if (report["min_cosine"] < 0.99999 or report["max_abs"] > 0.01 or
            report["max_logit_error"] > 0.005 or report["mean_top10_overlap"] < 9.5):
        raise RuntimeError("P0 merge numerical comparison failed; DO NOT EXPORT")
    print("P0 LORA MERGE NUMERICAL VALIDATION: PASS")
    return len(indices)


def check_onnx_with_ort(model, n: int, ort_n: int):
    import onnxruntime as ort
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise RuntimeError("Unexpected ONNX input/output count")
    errs, cosines, overlaps, logits = [], [], [], []
    for k in range(min(n, ort_n)):
        inp = np.load(OUT_DIR / f"case_{k:02d}_input.npy")
        ref = np.load(OUT_DIR / f"case_{k:02d}_pytorch_scores.npy")
        y = np.asarray(session.run([session.get_outputs()[0].name], {
            session.get_inputs()[0].name: inp})[0], dtype=np.float32)
        if y.shape != ref.shape or not np.isfinite(y).all():
            raise RuntimeError("Invalid ONNXRuntime output")
        cos = cosine(ref, y)
        diff = np.abs(ref.astype(np.float64) - y.astype(np.float64))
        overlap = len(set(np.argsort(-ref[0])[:10]) & set(np.argsort(-y[0])[:10]))
        logit_error = abs(topk_logit(ref)-topk_logit(y))
        cosines.append(cos); errs.append(float(diff.max()))
        overlaps.append(overlap); logits.append(logit_error)
        print(f"ORT case={k:02d} cosine={cos:.9f} max_abs={diff.max():.6f} "
              f"logit_err={logit_error:.6f} Top10={overlap}/10")
        np.save(OUT_DIR / f"case_{k:02d}_onnx_scores.npy", y)
    report = dict(samples=len(cosines), min_cosine=min(cosines), max_abs=max(errs),
                  max_logit_error=max(logits), mean_top10_overlap=float(np.mean(overlaps)))
    (OUT_DIR / "onnx_check.json").write_text(json.dumps(report, indent=2)+"\n")
    print("\nONNX Runtime report:", json.dumps(report, indent=2))
    if (report["min_cosine"] < 0.9999 or report["max_abs"] > 0.02 or
            report["max_logit_error"] > 0.01 or report["mean_top10_overlap"] < 9):
        raise RuntimeError("ONNX Runtime comparison failed")
    print("EARLY9 P0 PYTORCH -> ONNX: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["verify", "export"], default="verify")
    parser.add_argument("--num-check", type=int, default=12)
    parser.add_argument("--ort-check", type=int, default=4)
    args = parser.parse_args()

    if not CHECKPOINT.is_file() or not VAL_CACHE.is_file():
        raise FileNotFoundError("Missing P0 checkpoint or Early-9 val cache")
    if args.ort_check < 1:
        raise ValueError("--ort-check must be positive")
    print("EARLY-9 P0 TRAINED MODEL DEPLOYMENT")
    print("Phase:", args.phase)
    print("P0 checkpoint SHA256:", sha256_file(CHECKPOINT))
    print("Full pretrained tag:", training.MODEL_NAME)

    # Uses identical checkpoint-loading and source model code to the
    # successful P0 Eval8000 evaluation. CPU is intentional for export.
    reference, image_set, checkpoint = load_p0_reference(torch.device("cpu"))
    if reference.training:
        raise RuntimeError("Source P0 model must be in eval mode")
    deployment = EviRouteEarly9P0Deploy(reference, checkpoint).cpu().eval()
    paths, labels, indices = chosen_validation_samples(args.num_check)
    n = compare_pytorch_models(reference, deployment, image_set, paths, labels, indices)
    if args.phase == "verify":
        print("VERIFY COMPLETE. Rerun with --phase export after checking the log.")
        return

    import onnx
    import onnxruntime  # required: no silent skip of critical numerical validation
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dummy = np.load(OUT_DIR / "case_00_input.npy")
    with torch.inference_mode():
        torch.onnx.export(
            deployment, torch.from_numpy(dummy), str(ONNX_PATH),
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=["image"], output_names=["patch_scores"],
            dynamic_axes=None, dynamo=False,
        )
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("ONNX structure: PASS; size MiB:", round(ONNX_PATH.stat().st_size/1024**2, 3))
    check_onnx_with_ort(deployment, n, args.ort_check)
    meta = {
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_path": str(CHECKPOINT),
        "pretrained_model": training.MODEL_NAME,
        "num_blocks": 9,
        "feature_block_index": 8,
        "lora_merged_blocks": list(EXPECTED_LORA_BLOCKS),
        "lora_targets": list(EXPECTED_LORA_TARGETS),
        "input": "1x3x224x224 NCHW float32 RGB [0,1] bicubic",
        "output": "1x196 float32 patch scores (top10 is CPU postprocessing)",
        "onnx_sha256": sha256_file(ONNX_PATH),
        "torch_version": torch.__version__,
    }
    (OUT_DIR / "export_metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    print("ONNX MODEL READY:", ONNX_PATH)
    print("EARLY9 P0 END-TO-END EXPORT: PASS")


if __name__ == "__main__":
    main()
