# EviRoute-Edge
### Evidence-aware model adaptation and RK3588-oriented inference engineering

**Chenwei He (何晨炜)** · University of Electronic Science and Technology of China

EviRoute-Edge investigates how to retain **meaningful, less boundary-biased image evidence** while adapting a frozen vision model to a smaller, efficient inference path. It combines **DINOv3 ViT-B/16 Early-9**, selective **LoRA adaptation**, a **Top-K patch scorer**, and an experimentally controlled comparison between ordinary binary cross-entropy and EviRoute-inspired evidence debiasing.

The repository documents the training, model comparison, numerical audits and ONNX/RKNN FP16 export workflows. The related [EviRoute research repository](https://github.com/VIGO2002/EviRoute) describes the original evidence-routing study on DINOv3-7B; **these are separate architectures and experiment protocols**.

## Research-to-inference pipeline

```text
224×224 RGB image
        │
   DINOv3 ViT-B/16
   Early-9 blocks (0–8)
   LoRA: blocks 6–8, qkv/proj, r=8, α=8
        │
   Intermediate 14×14 patch features
   196 tokens × 768 dimensions
        │
   Normalization + patch scorer
   Linear(768,256) → GELU → Dropout → Linear(256,1)
        │
   196 evidence scores → Top-10 mean → fake logit
        │
        ├── R-BCE    : standard BCE objective
        └── R-Debias : training-only region-aware score dropout
                       + fake-only boundary evidence regularization
        │
   Frozen Step-2625 checkpoints
        │
   LoRA merge → External Patchify → ONNX → RKNN FP16
        │
   Per-image score, ranking and numerical alignment audits
```

The **debiased branch** masks scores before Top-K classification and penalizes outer-boundary evidence mass only for fake images. Inference uses **clean, unmasked scores**. The outermost patch ring B1 contains 52 of the 196 tokens; the second ring (B2-only) contains 44.

## Where to read the implementation

| Topic | Entry point | What it implements |
|---|---|---|
| Evidence debiasing | [`distillation/early9_p0_matched_pair_core.py`](distillation/early9_p0_matched_pair_core.py) | Ring masks, score dropout, fake-only edge penalty, BCE/debias branch losses |
| Strict matched-pair training | [`distillation/train_early9_p0_matched_pair.py`](distillation/train_early9_p0_matched_pair.py) | Shared initialization, source/checkpoint SHA guards, 2,625-step controlled comparison |
| DINOv3 Early-9 LoRA | [`distillation/train_early9_lora.py`](distillation/train_early9_lora.py) | Backbone construction, LoRA, patch scoring and original training protocol |
| Cross-generator evaluation | [`evaluation/eval_early9_matched_pair_genimage.py`](evaluation/eval_early9_matched_pair_genimage.py) | Fixed-checkpoint GenImage comparisons |
| Evidence faithfulness | [`evaluation/early9_faithfulness_v1_2/run_early9_rgb_faithfulness.py`](evaluation/early9_faithfulness_v1_2/run_early9_rgb_faithfulness.py) | Original-image perturbation against ring-matched random controls |
| Export and numerical audit | [`export/early9_pair_deployment_v1/export_pair.py`](export/early9_pair_deployment_v1/export_pair.py) | Frozen-checkpoint recovery, LoRA merge, external patchification and ONNX checks |
| RKNN FP16 build | [`export/early9_pair_deployment_v1/convert_pair_fp16.py`](export/early9_pair_deployment_v1/convert_pair_fp16.py) | RKNN-Toolkit2 model conversion and simulator validation |
| Comparison tools | [`scripts/early9_pair_480_tools_v1/`](scripts/early9_pair_480_tools_v1/) | Fixed-image packaging, numerical output comparison and source-wise analysis |

## Experiment snapshot

The **strict matched-pair** experiment uses **20,000 SD1.4 training images**, **600 source-validation images**, identical initial weights, and **2,625 updates** per arm. Both reported runs finish at step 2625. The training recipes share data order, optimizer and validation protocol; the intentional difference is the debiasing loss and stochastic token masking.

| Internal validation (600 images) | R-BCE | R-Debias | Difference |
|---|---:|---:|---:|
| AUC | 99.9633% | 99.9156% | −0.0478 pp |
| Fake Top-1 outer-ring B1 occupancy | 48.67% | 41.33% | **−7.33 pp** |
| Fake Top-10 outer-ring B1 occupancy | 43.23% | 39.23% | **−4.00 pp** |

**Exploratory cross-generator evaluation:** GenImage Eval8000 uses 8 generator groupings; the `unseen7` macro excludes training generator **SD1.4**.

| GenImage unseen-7 macro | R-BCE | R-Debias | Difference |
|---|---:|---:|---:|
| AUC | 85.62% | 85.17% | −0.45 pp |
| Fake Top-1 B1 occupancy | 41.94% | 38.97% | **−2.97 pp** |
| Fake Top-10 B1 occupancy | 39.17% | 37.56% | **−1.61 pp** |

These numbers are transcribed from the committed experiment summaries, not the results of a new run. **Eval8000 has already been inspected in previous experiments and is not a fresh independent blind test.** Its 8,000 logical rows reuse 500 real images across generator groups; it contains 4,500 unique images. Edge occupancy measures **where** selected evidence lies; it is not by itself an artifact localization or causal-faithfulness score. See [result interpretation](docs/RESULTS.md).

## Export workflow

The comparison preserves two distinct fixed-step checkpoints and source SHA identities. The deployment-oriented export path first audits LoRA merge and CPU external patchification, then builds a fixed-input ONNX model and uses **RKNN-Toolkit2 2.3.2** to produce an **RK3588-targeted FP16 RKNN** model (`do_quantization=False`).

The external patchify interface packs normalized RGB into `[1, 768, 1, 196]`, retaining the original 14×14 patch semantics. The included converter records source checkpoint, ONNX and RKNN SHA256 identities and per-sample numerical comparisons.

The committed conversion reports for **R-BCE** and **R-Debias** both record `simulator_pass`. This is a specific conversion/simulator finding, **not** an estimate of end-to-end image-processing latency. [Export protocol](export/early9_pair_deployment_v1/README.md).

## Repository structure

```text
EviRoute_edge/
├── distillation/    # Early-9 LoRA training and strict matched-pair experiments
├── evaluation/      # Generator transfer, perturbation and evidence diagnostics
├── export/          # LoRA merge, ONNX export, RKNN-Toolkit2 conversion
├── models/          # Patch-score inference modules
├── scripts/         # Benchmark preparation, numerical audits and utilities
├── tests/           # Algorithm/protocol tests and repository integrity checks
├── docs/            # Experiment interpretation, configuration, source manifest
├── tools/           # Source-integrity verification
├── requirements.txt
└── README.md
```

## Reproducing the source protocol

The scripts intentionally preserve the author's original paths and frozen SHA guards under `/root/autodl-tmp/EviRoute_edge`. Install a compatible PyTorch/CUDA stack and use the project's **FGTS** environment for its original training/evaluation workflows.

```bash
# From the original AutoDL project root:
cd /root/autodl-tmp/EviRoute_edge

# Verify that published text artifacts are unchanged after reorganization:
python tools/verify_source_manifest.py

# CPU algorithm tests (PyTorch required):
python -m unittest discover -s tests -p 'test_early9_p0_matched_pair.py' -v

# Inspect CLI without starting a model training run:
python distillation/train_early9_p0_matched_pair.py --help

# Paired-training protocol and prerequisites:
# docs/MATCHED_PAIR_20K.md
```

The original run depends on separately stored pretrained DINOv3 weights, historical checkpoints, image manifests, input images and score caches. These are intentionally excluded from the source repository. Do **not** bypass file-identity checks or present a structural/source-integrity test as a full training reproduction.

A complete protocol map, environment notes and measurement caveats are provided under [`docs/`](docs/).

## Authorship and attribution

**Chenwei He (何晨炜):** EviRoute-Edge research adaptation, controlled experiments, source implementation and numerical/export pipeline.

The project relies on third-party pretrained DINOv3 models, PyTorch, `timm`, RKNN-Toolkit2 and other scientific packages, each subject to its own distribution and license terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
