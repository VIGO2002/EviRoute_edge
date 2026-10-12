# Reproduction and source provenance

## Original environment

The experiment scripts retain their original local AutoDL root:

```text
/root/autodl-tmp/EviRoute_edge
```

They were collected from a historical research workspace rather than rewritten as an installable package. Using a different base path may trigger an intentional safety check in `distillation/train_early9_p0_matched_pair.py`.

For training/evaluation, the source uses PyTorch, timm, pandas, NumPy, Pillow and scikit-learn, with GPU-capable dependencies managed by the original `FGTS` environment. RKNN export is performed in a compatible **RKNN-Toolkit2 2.3.2** environment.

## Integrity and traceability

The archival [source-file manifest](SOURCE_FILE_MANIFEST.csv) lists the original files, SHA256 hashes and byte sizes. After restructuring, all manifest-listed original file contents remain unchanged; the original matched-pair protocol README is now `docs/MATCHED_PAIR_20K.md`.

Run:

```bash
python tools/verify_source_manifest.py
```

The checker reads the historical manifest and compares all listed files' SHA256 hashes, mapping only the moved protocol README to its new location. This checks **source identity**, not whether the experiments will run or produce identical results.

## External experiment inputs

Several critical artifacts are referred to by original paths and **are not checked into Git**:

- DINOv3 pretrained model weights
- Original base, shared-init and trained `*.pt` checkpoints
- Source and evaluation image manifests (such as the 20K training CSV and GenImage Eval8000 manifest)
- Frozen `*.npz` patch-score references
- Image data, extracted patch features, ONNX and RKNN model binaries

Missing artifacts should be restored from trusted local experiment storage. The identity guards should never be disabled solely to make a demonstration run.

## Protocol considerations

The matched-pair design retains a documented historical scorer dropout-mode behavior following intra-epoch evaluation, and its best-checkpoint selection is AUC-first with a BCE tie-break. Both R-BCE and R-Debias maintain this historical behavior to preserve the comparison.

The evaluation scripts contain experimental cohorts with repeated/shared real images and previous exploratory reuse. Do not claim the source repository alone establishes a new held-out benchmark result.

## Testing tiers

- **Static:** Python syntax compilation, file/source-manifest identity, directory and JSON schema inspection.
- **CPU smoke:** original torch-dependent tests in `tests/`, `evaluation/early9_*/*test*.py`, and `export/early9_pair_deployment_v1/test_pair_deployment.py`.
- **Full reproduction:** requires original checkpoints/datasets and historically matching AutoDL environments.

A passing static check never substitutes for full model equivalence.
