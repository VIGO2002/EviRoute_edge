# EviRoute-Edge | AutoDL to RK3588

**Author: Chenwei He (何晨炜)** · University of Electronic Science and Technology of China

**EviRoute-Edge** integrates DINOv3 ViT-B Early-9 LoRA adaptation, evidence-debiased training, rigorous paired experiments, and RK3588-targeted FP16 inference. The repository is organized into two **independent deployment roots**:

| Environment | Source partition | Core capabilities |
|---|---|---|
| **AutoDL / GPU / RKNN-Toolkit2** | [`server/`](server/) | 20K matched R-BCE vs R-Debias training, source/generalization evaluations, LoRA merge, ONNX export, RKNN FP16 conversion |
| **RK3588 / aarch64 / RKNNLite2** | [`rk3588/`](rk3588/) | Serial original-image inference, three-stage asynchronous pipeline, offline Web UI, evidence visualization, per-image audits |

## End-to-end architecture

```text
 AutoDL server                              RK3588 development board
 DINOv3 ViT-B Early-9                      Original JPEG/PNG images
 LoRA blocks 6–8, r=8                      RGB decode → CPU external patchify
 196 patch tokens, Top-10 MIL                    ↓
   ├─ R-BCE                                 RKNN FP16 NPU inference
   └─ R-Debias                              196 patch scores → Top-10
          ↓                                       ↓
  Paired evaluation & evidence audits       Fake/real score + evidence overlay
          ↓                                       ↓
  LoRA merge → ONNX → RKNN  ──────────────►  Serial or async inference
                                                   ↓
                                           Offline browser UI + reports
```

The R-Debias training objective applies region-aware patch-score dropout and a fake-only outer-boundary evidence penalty. Inference always uses clean patch scores.

## Inspect the actual implementation

| Focus | Source |
|---|---|
| Region-aware score masking and fake-only penalty | [`server/distillation/early9_p0_matched_pair_core.py`](server/distillation/early9_p0_matched_pair_core.py) |
| Controlled, same-initialization 20K training | [`server/distillation/train_early9_p0_matched_pair.py`](server/distillation/train_early9_p0_matched_pair.py) |
| Model export and LoRA merge | [`server/export/early9_pair_deployment_v1/export_pair.py`](server/export/early9_pair_deployment_v1/export_pair.py) |
| RKNN FP16 model conversion and simulation | [`server/export/early9_pair_deployment_v1/convert_pair_fp16.py`](server/export/early9_pair_deployment_v1/convert_pair_fp16.py) |
| **RK3588 serial inference** | [`rk3588/scripts/early9_full_pipeline_v1/run_images.py`](rk3588/scripts/early9_full_pipeline_v1/run_images.py) |
| **Bounded, three-stage asynchronous inference** | [`rk3588/scripts/early9_async_pipeline_v1/run_async.py`](rk3588/scripts/early9_async_pipeline_v1/run_async.py) |
| **Local offline inference service** | [`rk3588/scripts/eviroute_edge_ui_v1/server.py`](rk3588/scripts/eviroute_edge_ui_v1/server.py) |
| Browser front end | [`rk3588/scripts/eviroute_edge_ui_v1/static/`](rk3588/scripts/eviroute_edge_ui_v1/static/) |

## Original local paths and setup

**Important:** The `server/` and `rk3588/` folders are **GitHub source partitions**, not new filesystem roots for the deployed code. Copy their **contents** separately to their historical project roots:

| | Server | Board |
|---|---|---|
| Original project root | `/root/autodl-tmp/EviRoute_edge/` | `/root/EviRoute_RK3588/` |
| Dependencies | PyTorch, timm, CUDA, RKNN-Toolkit2 | RKNNLite2, NumPy, Pillow; Chromium for kiosk |
| Additional assets | DINOv3 weights, checkpoints, data manifests, image caches | Matching two `.rknn` binaries, RKNN runtime/driver |

The device's asynchronous inference script **imports functions from the serial pipeline at runtime**; keep `early9_async_pipeline_v1/` and `early9_full_pipeline_v1/` as siblings.

The offline UI defaults to loopback address `127.0.0.1:8765`. Do not publicly expose the service without access control.

See [installation and verification](docs/ENVIRONMENTS.md), [server research](server/README.md), and [board implementation](rk3588/README.md).

## Experimental evidence and limitations

The original matched-pair run used **20,000 SD1.4 training samples**, 600 held-out source-validation samples and 2,625 updates. Recorded validation fake Top-1 B1 occupancy fell **48.67% → 41.33%** (R-BCE → R-Debias); AUC was **99.96% → 99.92%**. In the **previously explored**, non-blind GenImage unseen-seven comparison, B1 Top-1 occupancy changed **41.94% → 38.97%** with Macro-AUC **85.62% → 85.17%**. These are historical logs, not re-measurements after restructuring; see [original server-side results](server/docs/RESULTS.md).

The uploaded board documentation reports approximately **10.6 images/s** under its 160-image, three-round asynchronous protocol with saved evidence outputs and around **1.51×** serial throughput. This is **historical throughput**, not new board testing, per-image latency or measured power.

**Source only:** Pretrained weights, large datasets, ONNX/RKNN binaries and private runtime jobs are not redistributed. Static integrity checks verify preserved source identity, not end-to-end deployability or numerical reproduction.

**Research and engineering implementation: Chenwei He (何晨炜).** Third-party DINOv3, PyTorch, timm and Rockchip components keep their original license terms; see [notices](THIRD_PARTY_NOTICES.md).
