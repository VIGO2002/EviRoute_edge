#!/usr/bin/env python3

import argparse
import json
import random
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from train_early9_lora import (
    ROOT,
    BASE_CKPT,
    EVAL_MANIFEST,
    Early9LoRA,
    ImageSet,
    load_data,
    check_features,
    evaluate,
    save_checkpoint,
    seed_everything,
)


OUT_ROOT = ROOT / "distillation/lora_runs"

HISTORICAL_TEST = Path(
    "/root/autodl-tmp/FGTS-main/features/"
    "genimage_sd14_smoke2k_mid20_patches/test.pt"
)


def canonical(p):
    return str(Path(str(p)).resolve())


class TrainingImages(ImageSet):
    def __init__(self, paths, labels, augmentation):
        super().__init__(paths, labels)
        self.augmentation = augmentation

    def __getitem__(self, idx):
        if self.augmentation == "none":
            # Reuse exactly the verified preprocessing.
            return super().__getitem__(idx)

        with Image.open(self.paths[idx]) as im:
            im = im.convert("RGB")
            im = im.resize(
                (224, 224),
                Image.Resampling.BICUBIC,
            )

            # Training-only resampling.
            if random.random() < 0.35:
                side = random.randint(112, 200)
                im = im.resize(
                    (side, side),
                    Image.Resampling.BICUBIC,
                )
                im = im.resize(
                    (224, 224),
                    Image.Resampling.BICUBIC,
                )

            # Training-only Gaussian blur.
            if random.random() < 0.30:
                im = im.filter(
                    ImageFilter.GaussianBlur(
                        radius=random.uniform(0.5, 1.5)
                    )
                )

            # Training-only JPEG compression.
            if random.random() < 0.35:
                buffer = io.BytesIO()
                im.save(
                    buffer,
                    format="JPEG",
                    quality=random.randint(50, 95),
                )
                buffer.seek(0)
                with Image.open(buffer) as compressed:
                    im = compressed.convert("RGB").copy()

            arr = np.asarray(
                im, dtype=np.float32
            ).copy()

        x = np.ascontiguousarray(
            (arr / 255.0).transpose(2, 0, 1)
        )

        return (
            torch.from_numpy(x),
            torch.tensor(
                float(self.labels[idx]),
                dtype=torch.float32,
            ),
        )


def prepare_training_set(
    manifest_path,
    original_set,
    validation_set,
    eval_paths,
):

    if manifest_path is None:
        return TrainingImages(
            original_set.paths,
            original_set.labels,
            augmentation="none",
        )

    manifest = pd.read_csv(manifest_path)

    required = {
        "path", "label", "origin"
    }

    if not required.issubset(manifest.columns):
        raise RuntimeError(
            f"Missing columns: {required - set(manifest.columns)}"
        )

    paths = [
        canonical(p)
        for p in manifest["path"]
    ]

    labels = manifest["label"].to_numpy(
        dtype=np.int64
    )

    if len(paths) != 20000:
        raise RuntimeError("Expected 20000 training images")

    if len(set(paths)) != len(paths):
        raise RuntimeError("Duplicate training paths")

    if not np.array_equal(
        np.bincount(labels, minlength=2),
        np.array([10000, 10000]),
    ):
        raise RuntimeError("Training class balance mismatch")

    if set(paths) & eval_paths:
        raise RuntimeError("Eval8000 training overlap")

    if set(paths) & set(validation_set.paths):
        raise RuntimeError("Validation training overlap")

    historical_test = torch.load(
        HISTORICAL_TEST,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    test_paths = {
        canonical(p)
        for p in historical_test["paths"]
    }

    if set(paths) & test_paths:
        raise RuntimeError("Historical test overlap")

    old_paths = set(original_set.paths)

    if not old_paths.issubset(set(paths)):
        raise RuntimeError(
            "Original 2800 samples not preserved"
        )

    origin_count = int(
        (manifest["origin"] == "original_2800").sum()
    )

    if origin_count != 2800:
        raise RuntimeError(
            "Original sample origin count mismatch"
        )

    if not all(Path(p).is_file() for p in paths):
        raise RuntimeError("Missing training images")

    print("Manifest:", manifest_path)
    print("Manifest samples:", len(paths))
    print("Original samples preserved:", len(old_paths))
    print("Historical test overlap: 0")
    print("Validation overlap: 0")
    print("Eval8000 overlap: 0")

    return paths, labels


def check_lora_gradients(model):
    total = 0.0

    for block in (6, 7, 8):
        attn = model.backbone.blocks[block].attn

        for target in ("qkv", "proj"):
            module = getattr(attn, target)
            grad = module.B.grad

            value = (
                0.0 if grad is None
                else float(grad.detach().abs().sum().item())
            )

            print(
                f"LoRA block={block} target={target} "
                f"B_grad_sum={value:.8e}"
            )

            if value <= 0 or not np.isfinite(value):
                raise RuntimeError(
                    f"Invalid LoRA gradient: {block}/{target}"
                )

            total += value

    print("Total LoRA B gradient:", total)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--run-name",
        default="D2_20k_noaug",
    )
    p.add_argument(
        "--train-manifest",
        type=str,
        default=None,
    )
    p.add_argument(
        "--augmentation",
        choices=["none", "classic"],
        default="none",
    )

    p.add_argument("--max-steps", type=int, default=2625)
    p.add_argument("--val-every", type=int, default=175)

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    if args.max_steps <= 0 or args.val_every <= 0:
        raise ValueError("Invalid training step configuration")

    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("Invalid loader configuration")

    if args.train_manifest is None and args.augmentation != "none":
        print("Using historical 2800 samples with augmentation")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    seed_everything(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda:0")

    output = OUT_ROOT / args.run_name
    output.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(EVAL_MANIFEST)
    eval_paths = {
        canonical(p)
        for p in eval_df["path"]
    }

    original_train, train_cache = load_data(
        "train", eval_paths
    )
    validation_set, _ = load_data(
        "val", eval_paths
    )

    if args.train_manifest is None:
        train_paths = original_train.paths
        train_labels = original_train.labels
    else:
        train_paths, train_labels = prepare_training_set(
            args.train_manifest,
            original_train,
            validation_set,
            eval_paths,
        )

    train_set = TrainingImages(
        train_paths,
        train_labels,
        args.augmentation,
    )

    print("=" * 72)
    print("EARLY-9 LORA DATA-SCALE EXPERIMENT")
    print("=" * 72)
    print("Train:", len(train_set))
    print("Val:", len(validation_set))
    print("Augmentation:", args.augmentation)
    print("Maximum optimizer steps:", args.max_steps)
    print("Validation interval:", args.val_every)
    print("Batch size:", args.batch_size)

    baseline = torch.load(
        BASE_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    model = Early9LoRA(
        baseline,
        rank=args.rank,
        alpha=args.alpha,
    ).to(device)

    # Verify with the original 2800-image cache,
    # independently of the new training manifest.
    check_features(
        model,
        original_train,
        train_cache,
        device,
    )

    trainable = [
        param
        for param in model.parameters()
        if param.requires_grad
    ]

    count = sum(p.numel() for p in trainable)

    print("Trainable parameters:", count)

    if count != 307713:
        raise RuntimeError(
            f"Unexpected trainable parameter count: {count}"
        )

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
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    initial = evaluate(model, val_loader, device)

    print("\nINITIAL VALIDATION")
    print(json.dumps(initial, indent=2))

    if abs(initial["auc"] - 0.9788888889) > 0.0005:
        raise RuntimeError(
            "Initial validation does not reproduce Early-7"
        )

    best = dict(initial)
    best_step = 0
    best_check = 0

    # Existing checkpoint format remains compatible.
    save_checkpoint(
        model, args, 0, best, output / "best.pt"
    )

    history = [{
        "validation_index": 0,
        "step": 0,
        "epoch": 0,
        "train_loss": None,
        **initial,
    }]

    global_step = 0
    epoch = 0
    validation_index = 0
    interval_losses = []

    while global_step < args.max_steps:
        epoch += 1

        model.train()
        model.backbone.eval()
        model.head.train()

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            logits, _, _ = model(images)

            loss = F.binary_cross_entropy_with_logits(
                logits, labels
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Nonfinite loss at step {global_step + 1}"
                )

            loss.backward()

            if global_step == 0:
                check_lora_gradients(model)

            torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=5.0,
            )

            optimizer.step()

            global_step += 1
            interval_losses.append(float(loss.item()))

            should_validate = (
                global_step % args.val_every == 0
                or global_step == args.max_steps
            )

            if should_validate:
                validation_index += 1
                metrics = evaluate(
                    model, val_loader, device
                )

                mean_loss = float(
                    np.mean(interval_losses)
                )
                interval_losses.clear()

                row = {
                    "validation_index": validation_index,
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    **metrics,
                }
                history.append(row)

                improved = (
                    metrics["auc"] > best["auc"] + 1e-6
                    or (
                        abs(metrics["auc"] - best["auc"]) <= 1e-6
                        and metrics["bce"] < best["bce"] - 1e-6
                    )
                )

                if improved:
                    best = dict(metrics)
                    best_step = global_step
                    best_check = validation_index

                    save_checkpoint(
                        model,
                        args,
                        epoch,
                        best,
                        output / "best.pt",
                    )

                print(
                    f"Step {global_step:04d}/{args.max_steps} "
                    f"epoch={epoch} "
                    f"train_loss={mean_loss:.6f} "
                    f"val_AUC={metrics['auc']*100:.4f}% "
                    f"val_BCE={metrics['bce']:.5f} "
                    f"B1={metrics['fake_top1_b1']*100:.2f}% "
                    f"{'*' if improved else ''}",
                    flush=True,
                )

            if global_step >= args.max_steps:
                break

    pd.DataFrame(history).to_csv(
        output / "history.csv",
        index=False,
    )

    summary = {
        "run": args.run_name,
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "completed_steps": global_step,
        "validation_count": validation_index,
        "augmentation": args.augmentation,
        "best_step": best_step,
        "best_validation_index": best_check,
        "best_val": best,
        "initial_val": initial,
        "selection": "maximum SD1.4 Val AUC; BCE tie-break",
        "config": vars(args),
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("DATA-SCALE LORA TRAINING COMPLETE")
    print("=" * 72)
    print("Completed steps:", global_step)
    print("Best step:", best_step)
    print("Best Val AUC:", f"{best['auc']*100:.4f}%")
    print("Best Val BCE:", f"{best['bce']:.6f}")
    print("Best Val B1:", f"{best['fake_top1_b1']*100:.2f}%")
    print("Checkpoint:", output / "best.pt")


if __name__ == "__main__":
    main()
