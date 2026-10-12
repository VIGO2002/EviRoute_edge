# Engineering architecture and implementation map

EviRoute-Edge distinguishes three layers of work:

1. **Algorithm adaptation:** ViT-B Early-9 features, LoRA blocks 6–8 and a token scorer; R-BCE and R-Debias share original initialization.
2. **Empirical audit:** Source-domain validation, GenImage transfer, input-space faithfulness and per-source evidence metrics.
3. **Export engineering:** Frozen checkpoint recovery, LoRA merge, external patchification, ONNX graph export and RKNN FP16 build/simulation.

## Algorithm source paths

- `distillation/train_early9_lora.py`: timm DINOv3 ViT-B creation, LoRA parameters, original dataset augmentation, model training.
- `distillation/train_early9_lora_data_v2.py`: 20K data source/manifest handling and original training data behavior.
- `distillation/early9_p0_matched_pair_core.py`: **main evidence debiasing module**. The `masks`, `masked_scores`, `edge_loss` and `compute_loss` functions are designed to be independently inspected/tested.
- `distillation/train_early9_p0_matched_pair.py`: verified shared initialization and two-arm matched training.

**Region definition:** 224×224 input, 16×16 patches, grid=14×14. B1 outermost ring has 52 positions; B2 is both outer rings, with 96 positions; B2-only contains 44, excluding B1.

**R-Debias loss:** take clean raw token scores `s` for each image. For classification, apply independent random token masking and additional B1/B2-only score masking *during training*, and use top-10 mean pooling before BCE. For fake images, compute a temperature-softmax of the **unmasked** `s` and penalize `ReLU(mass_B1 − ρ_B1)^2`, weighted by `λ_B1`. Current strict-pair values: global token dropout 0.15; B1 dropout 0.50; B2-only dropout 0.25; λ_B1 0.01; ρ_B1 0.40; λ_B2 0.

**Checkpoint selection:** the strict 20K pair intentionally uses the historical AUC-first, BCE-tiebreak rule in **both arms**. This is not interchangeable with other EviRoute checkpoint-selection variants.

## Export source paths

- `export/early9_pair_deployment_v1/pair_common.py`: frozen source and checkpoint identities, input shapes, Top-K numerical auditing.
- `export/early9_pair_deployment_v1/export_pair.py`: LoRA recovery and merge, PyTorch checks and ONNX export.
- `export/early9_pair_deployment_v1/convert_pair_fp16.py`: RKNN-Toolkit2 conversion (target `rk3588`, no integer quantization).
- `export/experiment_early9_external_patchify_onnx.py`: external patchify projection conversion.
- `scripts/early9_pair_480_tools_v1/prepare_autodl.py` / `analyze_autodl.py`: fixed-image preparation and numerical analysis.

A 224×224 RGB float32 input is packed to shape `[1,768,1,196]` (channels × patch pixels × patch positions), preserving a deterministic patch layout before the RKNN model's 1×1 projection. This interface is distinct from direct image input to a full vision backbone.

## Boundaries

Training/evaluation/export have different prerequisites and runtime dependencies. In particular, the RKNN conversion tool is not a replacement for a matching pretrained DINOv3 checkpoint or LoRA experimental run. Experimental conclusions require following their source-specific manifests and SHA guard protocol.
