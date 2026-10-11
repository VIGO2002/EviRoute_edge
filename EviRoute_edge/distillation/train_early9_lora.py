#!/usr/bin/env python3

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader


ROOT = Path("/root/autodl-tmp/EviRoute_edge")

FEATURE_DIR = ROOT / "distillation/depth_sweep_features/vitb_early9"
BASE_CKPT = ROOT / "distillation/depth_sweep_runs/early9/mid20_edge_debiased_mil_best.pt"
EVAL_MANIFEST = ROOT / "evaluation/genimage_8domain_eval8000/manifest.csv"
OUT_ROOT = ROOT / "distillation/lora_runs"

MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
LORA_BLOCKS = (6, 7, 8)
LORA_TARGETS = ("qkv", "proj")

DIM = 768
PATCHES = 196
TOPK = 10

BICUBIC = Image.Resampling.BICUBIC


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonical(path):
    return str(Path(str(path)).resolve())


class ImageSet(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            image = image.convert("RGB")
            image = image.resize((224, 224), BICUBIC)
            array = np.asarray(image, dtype=np.float32)

        array = np.ascontiguousarray(
            (array / 255.0).transpose(2, 0, 1)
        )

        return (
            torch.from_numpy(array),
            torch.tensor(float(self.labels[i]), dtype=torch.float32),
        )


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8.0):
        super().__init__()

        assert isinstance(base, nn.Linear)

        self.base = base
        self.rank = rank
        self.scale = alpha / rank

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.A = nn.Parameter(
            torch.empty(rank, base.in_features)
        )
        self.B = nn.Parameter(
            torch.zeros(base.out_features, rank)
        )

        nn.init.kaiming_uniform_(
            self.A, a=math.sqrt(5)
        )

    def forward(self, x):
        original = self.base(x)
        delta = F.linear(
            F.linear(x, self.A),
            self.B,
        )
        return original + delta * self.scale

    @property
    def weight(self):
        return self.base.weight + (self.B @ self.A) * self.scale

    @torch.no_grad()
    def merged_weight(self):
        return (
            self.base.weight
            + (self.B @ self.A) * self.scale
        )


class PatchScorer(nn.Module):
    def __init__(self):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(DIM, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.scorer(x).squeeze(-1)


class Early9LoRA(nn.Module):
    def __init__(self, baseline, rank, alpha):
        super().__init__()

        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=True,
        )

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        for block_idx in LORA_BLOCKS:
            attn = self.backbone.blocks[block_idx].attn

            for name in LORA_TARGETS:
                original = getattr(attn, name)

                setattr(
                    attn,
                    name,
                    LoRALinear(original, rank, alpha),
                )

        self.head = PatchScorer()

        self.head.load_state_dict(
            baseline["model_state_dict"],
            strict=True,
        )

        mu = baseline["mu"].float()
        std = baseline["std"].float()

        assert mu.shape == (DIM,)
        assert std.shape == (DIM,)
        assert (std > 0).all()

        self.register_buffer("mu", mu)
        self.register_buffer("std", std)

    def patches(self, images):
        b = self.backbone

        # Official timm EVA intermediate-feature API.
        # This handles positional embeddings, RoPE,
        # prefix tokens and stopping at block index 6.
        outputs = b.forward_intermediates(
            images,
            indices=[8],
            norm=False,
            stop_early=True,
            output_fmt="NLC",
            intermediates_only=True,
        )

        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise RuntimeError(
                f"Unexpected intermediate outputs: {type(outputs)}"
            )

        patches = outputs[0]

        expected = (images.shape[0], PATCHES, DIM)
        if tuple(patches.shape) != expected:
            raise RuntimeError(
                f"Unexpected patch shape: {tuple(patches.shape)}; "
                f"expected {expected}"
            )

        # Historical features were saved as FP16 and
        # converted back to FP32 before scoring.
        # Casting retains gradients during LoRA training.
        return patches.to(torch.float16).to(torch.float32)

    def forward(self, images):
        patches = self.patches(images)

        features = (
            patches - self.mu.reshape(1, 1, DIM)
        ) / self.std.reshape(1, 1, DIM)

        scores = self.head(features)

        values, indices = torch.topk(
            scores,
            k=TOPK,
            dim=1,
            sorted=True,
        )

        logits = values.mean(dim=1)

        return logits, scores, indices

    def lora_state(self):
        state = {}

        for idx in LORA_BLOCKS:
            attn = self.backbone.blocks[idx].attn

            for name in LORA_TARGETS:
                module = getattr(attn, name)
                prefix = f"blocks.{idx}.attn.{name}"

                state[prefix + ".A"] = (
                    module.A.detach().cpu().clone()
                )
                state[prefix + ".B"] = (
                    module.B.detach().cpu().clone()
                )

        return state


def load_data(split, eval_paths):
    path = FEATURE_DIR / f"{split}.pt"

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    paths = [canonical(p) for p in data["paths"]]
    labels = data["labels"].cpu().numpy().astype(np.int64)

    if len(paths) != len(set(paths)):
        raise RuntimeError(f"{split}: duplicate paths")

    if set(paths) & eval_paths:
        raise RuntimeError(f"{split}: Eval8000 overlap")

    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError(f"{split}: invalid labels")

    if not all(Path(p).is_file() for p in paths):
        raise RuntimeError(f"{split}: missing images")

    return ImageSet(paths, labels), data


@torch.inference_mode()
def check_features(model, dataset, cache, device):
    model.eval()

    images = torch.stack([
        dataset[i][0] for i in range(2)
    ]).to(device)

    computed = model.patches(images).cpu()
    reference = cache["patch_features"][:2].float()

    delta = (computed - reference).abs()

    mae = float(delta.mean())
    maximum = float(delta.max())

    print("\nHISTORICAL FEATURE ALIGNMENT")
    print(f"Feature MAE       : {mae:.8f}")
    print(f"Feature max error : {maximum:.8f}")

    if mae > 0.001 or maximum > 0.02:
        raise RuntimeError(
            "Manual Early-7 forward differs from cached features"
        )

    print("Feature alignment: PASS")


def edge_mask(indices):
    r = indices // 14
    c = indices % 14
    return (
        (r == 0) | (r == 13)
        | (c == 0) | (c == 13)
    )


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    all_y = []
    all_logits = []
    all_top10 = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        logits, _, indices = model(images)

        all_y.append(labels.numpy())
        all_logits.append(logits.cpu().numpy())
        all_top10.append(indices.cpu().numpy())

    y = np.concatenate(all_y).astype(np.int64)
    logits = np.concatenate(all_logits)
    top10 = np.concatenate(all_top10)

    fake_top10 = top10[y == 1]

    return {
        "auc": float(roc_auc_score(y, logits)),
        "ap": float(average_precision_score(y, logits)),
        "acc": float(np.mean((logits >= 0) == y)),
        "bce": float(
            F.binary_cross_entropy_with_logits(
                torch.from_numpy(logits),
                torch.from_numpy(y.astype(np.float32)),
            ).item()
        ),
        "fake_top1_b1": float(
            edge_mask(fake_top10[:, 0]).mean()
        ),
        "fake_top10_b1": float(
            edge_mask(fake_top10).mean()
        ),
    }


def save_checkpoint(model, args, epoch, metrics, output):
    checkpoint = {
        "head_state_dict": {
            k: v.detach().cpu().clone()
            for k, v in model.head.state_dict().items()
        },
        "lora_state_dict": model.lora_state(),
        "mu": model.mu.detach().cpu().clone(),
        "std": model.std.detach().cpu().clone(),
        "lora_config": {
            "rank": args.rank,
            "alpha": args.alpha,
            "blocks": list(LORA_BLOCKS),
            "targets": list(LORA_TARGETS),
        },
        "backbone": MODEL_NAME,
        "feature_block": 8,
        "best_epoch": epoch,
        "best_val_metrics": metrics,
        "base_checkpoint": str(BASE_CKPT),
        "training_args": vars(args),
    }

    torch.save(checkpoint, output)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", default="L1_lora_bce")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=0)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Invalid training parameters")

    seed_everything(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda:0")
    output = OUT_ROOT / args.run_name
    output.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(EVAL_MANIFEST)
    eval_paths = set(canonical(p) for p in manifest["path"])

    train_set, train_cache = load_data(
        "train", eval_paths
    )
    val_set, _ = load_data(
        "val", eval_paths
    )

    if set(train_set.paths) & set(val_set.paths):
        raise RuntimeError("Train/Val overlap")

    print("=" * 72)
    print("EARLY-7 END-TO-END LORA TRAINING")
    print("=" * 72)
    print("Train samples:", len(train_set))
    print("Val samples  :", len(val_set))
    print("LoRA blocks  :", LORA_BLOCKS)
    print("LoRA targets :", LORA_TARGETS)
    print("Rank / Alpha :", args.rank, args.alpha)
    print("Device       :", torch.cuda.get_device_name(0))

    baseline = torch.load(
        BASE_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    model = Early9LoRA(
        baseline,
        args.rank,
        args.alpha,
    ).to(device)

    check_features(model, train_set, train_cache, device)

    trainable = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    lora_count = sum(
        p.numel() for n, p in trainable
        if ".A" in n or ".B" in n
    )

    head_count = sum(
        p.numel() for n, p in trainable
        if n.startswith("head.")
    )

    total = sum(p.numel() for _, p in trainable)

    print("\nTRAINABLE PARAMETERS")
    print("LoRA:", lora_count)
    print("Head:", head_count)
    print("Total:", total)

    if lora_count == 0 or head_count == 0:
        raise RuntimeError("No LoRA or scorer parameters")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    optimizer = torch.optim.AdamW(
        [p for _, p in trainable],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    initial = evaluate(model, val_loader, device)

    print("\nINITIAL BASELINE")
    print(json.dumps(initial, indent=2))

    # This must reproduce the known Early-7 baseline.
    if abs(initial["auc"] - 0.9788889) > 0.0005:
        raise RuntimeError(
            "Epoch-0 AUC does not match Early-7 reference"
        )

    best = copy.deepcopy(initial)
    best_epoch = 0

    save_checkpoint(
        model, args, 0, best, output / "best.pt"
    )

    history = [{
        "epoch": 0,
        "train_loss": None,
        **initial,
    }]

    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Keep original ViT blocks in inference mode.
        # Only LoRA matrices receive backbone gradients.
        model.backbone.eval()
        model.head.train()

        losses = []
        steps = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, _, _ = model(images)

            loss = F.binary_cross_entropy_with_logits(
                logits, labels
            )

            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")

            loss.backward()

            if epoch == 1 and steps == 0:
                grad_sum = 0.0
                for idx in LORA_BLOCKS:
                    attn = model.backbone.blocks[idx].attn
                    for target in LORA_TARGETS:
                        module = getattr(attn, target)
                        grad = module.B.grad
                        g = (
                            0.0 if grad is None
                            else float(grad.detach().abs().sum().item())
                        )
                        print(
                            f"LoRA block={idx} target={target} "
                            f"B_grad_sum={g:.8e}"
                        )
                        if g <= 0:
                            raise RuntimeError(
                                f"LoRA gradients missing: {idx}/{target}"
                            )
                        grad_sum += g

                print("LoRA B gradient sum:", grad_sum)

            nn.utils.clip_grad_norm_(
                [p for _, p in trainable],
                max_norm=5.0,
            )

            optimizer.step()

            losses.append(float(loss.item()))
            steps += 1

            if (
                args.max_train_steps > 0
                and steps >= args.max_train_steps
            ):
                break

        metrics = evaluate(
            model, val_loader, device
        )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **metrics,
        }
        history.append(row)

        # Highest Val AUC; use BCE only as a tie breaker.
        improved = (
            metrics["auc"] > best["auc"] + 1e-6
            or (
                abs(metrics["auc"] - best["auc"]) <= 1e-6
                and metrics["bce"] < best["bce"] - 1e-6
            )
        )

        if improved:
            best = copy.deepcopy(metrics)
            best_epoch = epoch
            save_checkpoint(
                model, args, epoch, best,
                output / "best.pt"
            )
            stale = 0
        else:
            stale += 1

        print(
            f"Epoch {epoch:02d} "
            f"loss={row['train_loss']:.5f} "
            f"val_AUC={metrics['auc']*100:.4f}% "
            f"val_BCE={metrics['bce']:.5f} "
            f"B1={metrics['fake_top1_b1']*100:.2f}% "
            f"{'*' if improved else ''}",
            flush=True,
        )

        if stale >= args.patience:
            print("Early stopping:", epoch)
            break

    pd.DataFrame(history).to_csv(
        output / "history.csv",
        index=False,
    )

    summary = {
        "initial_val": initial,
        "best_epoch": best_epoch,
        "best_val": best,
        "selection": "maximum SD1.4 Val AUC, BCE tie break",
        "config": vars(args),
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("EARLY-7 LORA TRAINING COMPLETE")
    print("=" * 72)
    print("Best epoch:", best_epoch)
    print("Best Val AUC:", f"{best['auc']*100:.4f}%")
    print("Best Val BCE:", f"{best['bce']:.6f}")
    print("Best Val B1:", f"{best['fake_top1_b1']*100:.2f}%")
    print("Checkpoint:", output / "best.pt")


if __name__ == "__main__":
    main()
