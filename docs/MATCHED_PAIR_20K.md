# Early-9 P0 strict 20k matched pair: R-BCE / R-Debias

## Purpose

Test whether applying original EviRoute region-aware score dropout plus fake-only outer-boundary mass loss **from step 1** changes the spatial evidence trajectory when otherwise matching the actual historical P0 recipe. Test is *not* the 3-epoch 2.8k ablation, and is *not* P0 fine-tuning. It is two independent full trainings from one shared seed-42 initialization.

**Never select configurations using the already inspected AIGIBench pilot200 or GenImage Eval8000.** This experiment uses only the historical 20k training manifest and 600 original validation images. The project does not change original model or data source files.

## Audited source contracts

- `/root/autodl-tmp/EviRoute_edge/distillation/train_early9_lora.py` sha256 `e31e0b7154d99efa4de2d1ca723b50940dd697e56590f627d21cda145f4dc9ba`
- `/root/autodl-tmp/EviRoute_edge/distillation/train_early9_lora_data_v2.py` sha256 `e417dd8c6be9c792c8d802bbe1e9eb2c8678f43a175d56ebac663d843841760b`
- Historical P0 sha256 `79f1983e83d905e0e26f2f64ec181f1800dcff80d399e31341c29c52a0a2e995` **for recipe-parameter verification only**, NOT used to initialize either arm. The historical P0 `training_args` are loaded read-only to assert actual workers and all other recorded settings.
- Pre-P0 head / feature mu,std from `distillation/depth_sweep_runs/early9/mid20_edge_debiased_mil_best.pt`; fresh LoRA A Kaiming, LoRA B zeros, seed42; ViT-B block 0-8, train LoRA blocks6-8 qkv/proj + scorer.
- `distillation/data_manifests/sd14_train20k_seed20261008.csv` 20k (10k/10k), preserves original 2800; the old data validator rejects Eval8000, original val and historical test overlaps. Source file, base, train manifest, train/val order SHA identities recorded and checked against shared init.
- Original: AdamW on all 307713 trainable parameters with **single fixed LR=1e-4**, WD=1e-4, grad clip 5, batch16, seed42, max2625 steps, validation every175, and *classic* training-only resampling/blur/JPEG augmentation. No scheduler. `workers=4` is the source CLI default. A hard preflight guard cross-checks the historical P0 checkpoint `training_args["workers"]` and stops if the actual value differs; rerun with the historical `--workers N` after confirming it.
- Validation: original Top10 mean logits, no dropout/masking, AUC primary selection, BCE tie-break. Save best.pt AND final_step.pt (2625); 16 validation entries (epoch0 + 15).
- **Historical quirk preserved intentionally**: `evaluate()` invokes `model.eval()`; training restores scorer `.train()` only at next epoch beginning. So after each mid-epoch validation the scorer's 0.1 dropout remains off until new epoch. Both arms keep this behavior. It is not scientifically ideal, but must be matched for P0 reproduction.
- R-Debias only difference: EviRoute `token_dropout=.15`, `edge1_dropout=.5`, `edge2_only_dropout=.25` on patch *scores* for training BCE; use **unmasked** scores for fake-only softmax edge hinge² penalty λ=.01, ρ=.4, τ=1; λ_B2=0. R-BCE uses original BCE(logits, labels) directly.
- EviRoute dropout consumes CUDA RNG and changes gradient stochasticity relative to R-BCE by design. The loader's own seeded `torch.Generator` makes sample ordering/worker seeds identical across runs when environment and workers match; augmentation is still randomized by DataLoader workers and exact bitwise parity across independent GPU runs is not guaranteed.

## Install and test

Upload ZIP to project root, then:

```bash
cd /root/autodl-tmp/EviRoute_edge
conda activate FGTS
unzip -n early9_p0_matched_pair_20k_v1.zip -d .
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
python -m unittest discover -s tests -p 'test_early9_p0_matched_pair.py' -v
```

## One-time shared initialization (requires GPU)

**Preflight:** the script refuses to proceed if `P0_Early9_20k_classic/best.pt` SHA or its recorded `training_args` do not match. If workers differ from 4, pass `--workers <historical value>` consistently to all commands; do not alter model files.

```bash
python -u distillation/train_early9_p0_matched_pair.py --stage init --workers 4
```

Expected `INIT VAL` AUC within 0.0005 of .9788888889, 600 val, model alignment PASS and `P0 MATCHED PAIR INITIALIZATION: PASS`. Saves under `distillation/p0_matched_pair_20k_v1/shared_seed42_init.pt` (never overwritten).

## Dry-run each training arm (no weight update, no output directory)

```bash
python -u distillation/train_early9_p0_matched_pair.py --stage train --arm R-BCE --workers 4 --dry-run
python -u distillation/train_early9_p0_matched_pair.py --stage train --arm R-Debias --workers 4 --dry-run
```

Both should say `P0 MATCHED PAIR DRY RUN: PASS` and positive LoRA B gradients.

## Stage 1: reproduce historical P0 with R-BCE

```bash
mkdir -p logs
set -o pipefail
python -u distillation/train_early9_p0_matched_pair.py --stage train --arm R-BCE --workers 4 \
  2>&1 | tee logs/p0_matched_R_BCE_v1.log
cat distillation/p0_matched_pair_20k_v1/R_BCE/summary.json
```

Historical comparison targets: best val AUC≈.9996333, Fake Top1 B1≈.486667, Top10 B1≈.432333. Last-checkpoint best_step historically2625. These are *diagnostic reference values*, not assertions of exact bitwise replay. Check discrepancy before proceeding. Under the original training-mode quirk, expect approximately 450 head training steps and 2175 head eval-mode training steps.

## Stage 2: R-Debias (ONLY after R-BCE reproduction is reviewed)

```bash
python -u distillation/train_early9_p0_matched_pair.py --stage train --arm R-Debias --workers 4 \
  2>&1 | tee logs/p0_matched_R_Debias_v1.log
cat distillation/p0_matched_pair_20k_v1/R_Debias/summary.json
python -u evaluation/summarize_early9_p0_matched_pair.py
```

Comparison: **fixed Step2625** and **best validation-AUC model**, at all 15 shared validation steps. If R-Debias differs, measure both AUC retention and fake B1/B2 counts. Compare source-level generalization only on independent data that were not used for method tuning. Edge heatmap location is not a faithfulness test.

## Safeguards and limitations

- Existing directories abort new run: do not delete or overwrite old runs casually. All outputs are isolated from P0 and previous Debias studies.
- The *original* model's pre-P0 feature-cache alignment test is called before any training; input processing, model construction, and dataset augmentation are imported directly from historical original source files.
- Baseline is loaded from the historical base checkpoint, **not the trained P0 best.pt**.
- `--dry-run` performs one backward pass but no `optimizer.step` and no output files.
- The source SHA guard requires the exact historical scripts unchanged. If your files differ, stop and investigate rather than removing the guard.
- If historical P0 training used nondefault worker count, compare/re-run using that count; random augmentation depends on worker seed distribution.
- Formal inference does not use any training masks, only raw patch-score Top10 mean; the exact historical checkpoint selection AUC+BCE rule is shared by both arms, deliberately not the original EviRoute edge-aware selection.
