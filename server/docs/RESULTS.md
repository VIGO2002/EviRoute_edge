# Evidence, generalization and numerical comparison

All numeric claims below are **read from JSON/CSV results already included in the source collection**. They are not fresh reproduction measurements.

## Experimental isolation

The 20,000-image source-only training compares two models from a shared, explicitly verified initialization (`seed=42`). Both arms use the historical 2,625-step optimizer and augmentation schedule with 600 source-validation images. The only intended treatment is adding EviRoute region-aware score dropout and a fake-only outer-boundary mass penalty to R-Debias. Stochastic dropout necessarily consumes extra random numbers, so independently executed GPU runs are not bitwise identical.

References:
- [R-BCE training report](../distillation/p0_matched_pair_20k_v1/R_BCE/summary.json)
- [R-Debias training report](../distillation/p0_matched_pair_20k_v1/R_Debias/summary.json)
- [Paired training comparison](../distillation/p0_matched_pair_20k_v1/paired_summary.json)

### Source-validation measurements

| Metric | R-BCE | R-Debias |
|---|---:|---:|
| AUC | 0.9996333 | 0.9991556 |
| AP | 0.9996358 | 0.9991327 |
| Accuracy | 0.9900 | 0.9833 |
| Fake B1 Top-1 | 0.4867 | 0.4133 |
| Fake B1 Top-10 | 0.4323 | 0.3923 |

No claim is made that the performance change is statistically significant based on this table alone. The original training protocol retains a historical head-dropout train/eval-mode quirk, documented in [matched-pair notes](MATCHED_PAIR_20K.md).

## Cross-generator transfer: GenImage Eval8000

Reference: [strict comparison summary](../evaluation/early9_RBCE_vs_RDebias_genimage8000_v1/summary.json).

### Seven unseen source macro

| Metric | R-BCE | R-Debias | R-Debias minus R-BCE |
|---|---:|---:|---:|
| Macro AUC | 0.8561737 | 0.8516571 | −0.0045166 |
| Fake B1 Top-1 occupancy | 0.4194286 | 0.3897143 | −0.0297143 |
| Fake B1 Top-10 occupancy | 0.3916857 | 0.3756000 | −0.0160857 |

SD1.4 is the source generator excluded from the `unseen7` view. `all8` includes SD1.4.

### Statistical interpretation

- Eval8000 contains 8,000 source-grouped rows but only 4,500 unique images; eight groups reuse the same 500 real images.
- The selected models were fixed before this evaluation, but the evaluated image set had been explored previously and **is not a new final blind test**.
- Lower B1/B2 occupancy does **not**, on its own, prove evidence faithfulness, correct forgery localization or absence of shortcut features.
- Source-wise changes are not uniform. Report per-generator effects rather than claiming improvement on every generator.
- Avoid treating repeated real-image rows as independent observations in significance analysis.

## Original-image faithfulness

Reference: [240-image experiment](../evaluation/early9_rgb_faithfulness_240_v1_2/summary.json).

The faithfulness protocol perturbs selected **16×16 RGB patches** toward each patch's local mean and compares the resulting logit changes to exact ring-matched random-patch controls. On the seven unseen fake-image groups (210 samples), the original logs record:

| Metric | R-BCE | R-Debias |
|---|---:|---:|
| Perturbation advantage | 1.6491 | 1.5936 |
| Win rate against ring-matched random controls | 91.43% | 92.86% |

These are separate within-model diagnostic results and do not mean R-Debias has a greater perturbation *magnitude* than R-BCE. The sample set is exploratory.

## RKNN export/simulator provenance

References:
- [R-BCE FP16 conversion report](../export/early9_pair_deployment_v1/R_BCE/rk3588_fp16/conversion_validation.json)
- [R-Debias FP16 conversion report](../export/early9_pair_deployment_v1/R_Debias/rk3588_fp16/conversion_validation.json)

Both JSONs record `status: simulator_pass`, `passed: true` for RK3588-targeted FP16 conversion using RKNN-Toolkit2 2.3.2, with numerical reference cases. This does not by itself establish UI throughput, disk-to-screen latency, temperature stability or energy consumption.

## Additional numerical alignment records

The repository includes a fixed 480-image paired evaluation and [analysis summary](../evaluation/early9_pair_480_analysis_v1/summary.json). This is a previously analyzed cohort, distinct from Eval8000's 8,000 logical rows. The 480-image analysis reports PyTorch/target-runtime numerical and ranking comparisons, and should be interpreted as an equivalence audit, **not a generalization benchmark or fresh causal evidence evaluation**.
