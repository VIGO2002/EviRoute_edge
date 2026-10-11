#!/usr/bin/env python3
"""Evaluate Early-9 P0 LoRA on existing GenImage Eval8000.

Uses the proven Early-9 LoRA model implementation and exact historical
ImageSet preprocessing. No test-set checkpoint selection is performed.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path('/root/autodl-tmp/EviRoute_edge')
DIST = ROOT / 'distillation'
BENCH = ROOT / 'evaluation/genimage_8domain_eval8000'
VAL_CACHE = ROOT / 'distillation/depth_sweep_features/vitb_early9/val.pt'
P0_CKPT = ROOT / 'distillation/lora_runs/P0_Early9_20k_classic/best.pt'
OUT = ROOT / 'evaluation/genimage_8domain_early9_P0'
DOMAINS = ['ADM', 'BigGAN', 'Midjourney', 'VQDM', 'GLIDE', 'SD1.4', 'SD1.5', 'Wukong']
HARD = ['ADM', 'BigGAN', 'VQDM', 'GLIDE']


def boundary_masks(indices):
    rows = indices // 14
    cols = indices % 14
    b1 = (rows == 0) | (rows == 13) | (cols == 0) | (cols == 13)
    b2 = (rows <= 1) | (rows >= 12) | (cols <= 1) | (cols >= 12)
    return b1, b2


def load_model(device):
    sys.path.insert(0, str(DIST))
    module = importlib.import_module('train_early9_lora')
    candidates = [getattr(module, name, None) for name in ('Early9LoRA', 'Early7LoRA')]
    cls = next((x for x in candidates if isinstance(x, type)), None)
    if cls is None:
        raise RuntimeError('No Early9LoRA/Early7LoRA class in train_early9_lora.py')
    if not hasattr(module, 'ImageSet') or not hasattr(module, 'BASE_CKPT'):
        raise RuntimeError('train_early9_lora.py lacks ImageSet or BASE_CKPT')

    baseline = torch.load(module.BASE_CKPT, map_location='cpu', weights_only=False)
    ckpt = torch.load(P0_CKPT, map_location='cpu', weights_only=False)
    cfg = ckpt['lora_config']
    blocks = list(map(int, cfg['blocks']))
    targets = list(cfg['targets'])
    if blocks != [6, 7, 8] or targets != ['qkv', 'proj']:
        raise RuntimeError(f'Unexpected LoRA configuration: {cfg}')
    rank, alpha = int(cfg['rank']), float(cfg['alpha'])
    model = cls(baseline, rank=rank, alpha=alpha).to(device)
    if not torch.equal(model.mu.cpu(), ckpt['mu'].float()):
        raise RuntimeError('mu buffer mismatch')
    if not torch.equal(model.std.cpu(), ckpt['std'].float()):
        raise RuntimeError('std buffer mismatch')

    model.head.load_state_dict(ckpt['head_state_dict'], strict=True)
    target_weights = ckpt['lora_state_dict']
    keys_expected = {
        f'blocks.{block}.attn.{target}.{part}'
        for block in blocks for target in targets for part in ('A', 'B')
    }
    if set(target_weights) != keys_expected:
        raise RuntimeError(f'LoRA key mismatch: {set(target_weights) ^ keys_expected}')
    total_B = 0.0
    with torch.no_grad():
        for key, value in target_weights.items():
            parts = key.split('.')
            block = int(parts[1]); target = parts[3]; ab = parts[4]
            dest = getattr(getattr(model.backbone.blocks[block].attn, target), ab)
            if dest.shape != value.shape:
                raise RuntimeError(f'LoRA parameter shape mismatch for {key}')
            dest.copy_(value.to(device=dest.device, dtype=dest.dtype))
            if ab == 'B':
                total_B += float(dest.detach().abs().sum().item())
    if not np.isfinite(total_B) or total_B <= 0:
        raise RuntimeError('LoRA weights did not load correctly')
    model.eval()
    print('EARLY-9 checkpoint:', P0_CKPT)
    print('Checkpoint best epoch:', ckpt.get('best_epoch'))
    print('LoRA |B| sum:', total_B)
    print('Checkpoint loading: PASS')
    return model, module.ImageSet, ckpt


@torch.inference_mode()
def inference(model, dataset_cls, paths, device, batch_size, workers):
    ds = dataset_cls(paths, np.zeros(len(paths), dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    logits_all, scores_all, top10_all = [], [], []
    for batch_i, (images, _) in enumerate(loader, 1):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        logits, scores, indices = model(images)
        if scores.ndim != 2 or scores.shape[1] != 196 or indices.shape[1] != 10:
            raise RuntimeError(f'Bad output shapes: {scores.shape}, {indices.shape}')
        logits_all.append(logits.cpu().numpy().astype(np.float32))
        scores_all.append(scores.cpu().numpy().astype(np.float32))
        top10_all.append(indices.cpu().numpy().astype(np.int16))
        n_done = min(batch_i * batch_size, len(ds))
        if n_done % 400 < batch_size or n_done == len(ds):
            print(f'Inference {n_done}/{len(ds)}', flush=True)
    z = np.concatenate(logits_all)
    scores = np.concatenate(scores_all)
    top10 = np.concatenate(top10_all)
    if z.shape != (len(paths),) or not np.isfinite(z).all() or not np.isfinite(scores).all():
        raise RuntimeError('Nonfinite or incorrect inference outputs')
    if np.any(top10 < 0) or np.any(top10 > 195):
        raise RuntimeError('Invalid patch indices')
    return z, scores, top10


def domain_metrics(labels, logits, indices):
    fake = indices[labels == 1]
    b1, b2 = boundary_masks(fake)
    return {
        'auc': float(roc_auc_score(labels, logits)),
        'ap': float(average_precision_score(labels, logits)),
        'acc_at_zero': float(np.mean((logits >= 0) == labels)),
        'fake_top1_B1': float(b1[:, 0].mean()),
        'fake_top10_B1': float(b1.mean()),
        'fake_top1_B2': float(b2[:, 0].mean()),
        'fake_top10_B2': float(b2.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError('Invalid DataLoader arguments')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0')
    OUT.mkdir(parents=True, exist_ok=True)

    model, dataset_cls, ckpt = load_model(device)

    # Reproducibility gate: the restored best.pt MUST reproduce its logged Val AUC.
    val = torch.load(VAL_CACHE, map_location='cpu', weights_only=False, mmap=True)
    val_paths = [str(x) for x in val['paths']]
    val_labels = torch.as_tensor(val['labels']).cpu().numpy().astype(np.int64)
    val_logits, _, _ = inference(model, dataset_cls, val_paths, device, args.batch_size, args.workers)
    val_auc = float(roc_auc_score(val_labels, val_logits))
    print('Restored Val AUC:', f'{val_auc*100:.4f}%')
    expected_auc = float(ckpt['best_val_metrics']['auc'])
    if abs(val_auc - expected_auc) > 0.00001:
        raise RuntimeError(f'Val AUC mismatch: restored={val_auc}, saved={expected_auc}')
    print('Checkpoint validation reproduction: PASS')

    df = pd.read_csv(BENCH / 'manifest.csv').sort_values('global_index').reset_index(drop=True)
    if len(df) != 8000 or df['global_index'].tolist() != list(range(8000)):
        raise RuntimeError('Unexpected Eval8000 manifest indexing')
    if set(df['domain']) != set(DOMAINS):
        raise RuntimeError('Unexpected Eval8000 domains')
    for name in DOMAINS:
        part = df[df['domain'] == name]
        if len(part) != 1000 or set(part['label'].value_counts().to_dict().values()) != {500}:
            raise RuntimeError(f'Unexpected domain balance for {name}')

    unique_paths = df['path'].drop_duplicates().tolist()
    mapping = {p: idx for idx, p in enumerate(unique_paths)}
    gather = np.array([mapping[p] for p in df['path']], dtype=np.int64)
    print('Eval8000 unique images:', len(unique_paths))
    if len(unique_paths) != 4500:
        raise RuntimeError('Unexpected count of unique Eval8000 paths')
    unique_logits, unique_scores, unique_top10 = inference(
        model, dataset_cls, unique_paths, device, args.batch_size, args.workers)
    logits, scores, top10 = unique_logits[gather], unique_scores[gather], unique_top10[gather]

    results = []
    for domain in DOMAINS:
        mask = df['domain'].to_numpy() == domain
        y = df.loc[mask, 'label'].to_numpy(dtype=np.int64)
        metrics = domain_metrics(y, logits[mask], top10[mask])
        results.append({'domain': domain, **metrics})
    frame = pd.DataFrame(results)
    print('\n' + '=' * 72)
    print('P0 EARLY-9 PER-DOMAIN AUC (%)')
    print('=' * 72)
    shown = frame[['domain', 'auc', 'ap', 'fake_top1_B1', 'fake_top10_B1']].copy()
    for col in shown.columns[1:]: shown[col] = shown[col] * 100
    print(shown.round(3).to_string(index=False))

    unseen = frame[frame['domain'] != 'SD1.4']
    hard = frame[frame['domain'].isin(HARD)]
    summary = {
        'status': 'exploratory',
        'model': 'P0_Early9_20k_classic',
        'checkpoint': str(P0_CKPT),
        'best_epoch': ckpt.get('best_epoch'),
        'restored_val_auc': val_auc,
        'unseen7_macro_auc': float(unseen['auc'].mean()),
        'unseen7_worst_auc': float(unseen['auc'].min()),
        'unseen7_worst_domain': str(unseen.loc[unseen['auc'].idxmin(), 'domain']),
        'difficult4_macro_auc': float(hard['auc'].mean()),
        'unseen7_fake_top1_B1': float(unseen['fake_top1_B1'].mean()),
        'unseen7_fake_top10_B1': float(unseen['fake_top10_B1'].mean()),
    }
    print('\n' + '=' * 72)
    print('GENERALIZATION AND EVIDENCE')
    print('=' * 72)
    print('Unseen-7 Macro AUC:', f"{summary['unseen7_macro_auc']*100:.3f}%")
    print('Worst AUC:', f"{summary['unseen7_worst_auc']*100:.3f}%",
          'domain:', summary['unseen7_worst_domain'])
    print('Difficult-4 Macro AUC:', f"{summary['difficult4_macro_auc']*100:.3f}%")
    print('Unseen-7 fake Top-1 B1:', f"{summary['unseen7_fake_top1_B1']*100:.2f}%")
    print('Unseen-7 fake Top-10 B1:', f"{summary['unseen7_fake_top10_B1']*100:.2f}%")

    frame.to_csv(OUT / 'per_domain_metrics.csv', index=False)
    np.savez_compressed(OUT / 'p0_outputs.npz',
                        global_index=df['global_index'].to_numpy(dtype=np.int64),
                        logits=logits, patch_scores=scores, top10_indices=top10)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nSaved:', OUT)
    print('P0 EARLY-9 CROSS-GENERATOR EVALUATION: PASS')


if __name__ == '__main__':
    main()
