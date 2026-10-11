# Early-9 R-BCE / R-Debias Detail4 Characterization V1

## 目的与实验性质

输入空间证据的 **low-level characterization（低层视觉特征描述）**，与之前完成的 RGB perturbation faithfulness（判别忠实性）互补，但两者不等价。

冻结的两组模型为 `distillation/p0_matched_pair_20k_v1/{R_BCE,R_Debias}/final_step.pt`（Step 2625）。不训练，不调整权重。本阶段的 GenImage Eval8000 曾被多次分析，是**探索性分析，不是新的独立盲测**。

### 关键定义（请严格区分）

对每张 224×224 RGB 图分成 196 个 16×16 图像块，计算：

- `lap_var`：Laplacian 局部方差；
- `grad_energy`：灰度图有限差分梯度能量；
- `residual_abs`：原图灰度与 GaussianBlur(radius=1) 灰度的残差绝对值均值；
- `color_var`：RGB 三通道局部方差均值；
- `fft_hf_ratio`：FFT 高频能量占比，频域半径门槛 0.35。

上面五项 **完全从用户提供的原版 `analyze_lowlevel_red_blue_v2.py` 复用 `compute_patch_metrics` 与 `percentile_ranks`**。重要：这份原版代码实际把 **5 项**取平均，即 `legacy5_percentile_delta`。为了回应目前提出的“四项 Detail4”，本实验额外并列报告明确指定的 **4 项（Lap / Grad / Residual / Color）**平均值 `detail4_percentile_delta`。**二者不能混称。**

对每张图，在整张图的 196 个 patch 里分别计算各视觉统计量的百分位秩，然后计算：

`contrast = mean(percentile(red patches)) - mean(percentile(blue patches))`

- Fake 的 red = 当前冻结的 **该模型自身** `paired_outputs.npz` Top-10 高 fake-support 区域。
- Real 的 red = `-fake_patch_scores` 的前 10 个，即 **低 fake-support / 高 real-aligned support** 区域。
- blue = 同环（B1 最外圈、B2-only 第二圈、Interior 内部）且不与 red 重叠的 **最低 class-aligned support** patch，精确匹配 red 各环数量；不是任意不匹配低分区域。

红蓝之间的负对比表示高支持区域的这些视觉量在**该图内部相对更低**，正对比反之。**Real 的 class-aligned 选区仅作描述；由于 Early-9 图像 Logit 来自最高 fake-support Top-10，低 fake-support 位置并非直接参与 Top-10 聚合，不能冒称已经证明 real decision faithfulness。**

## 准备环境

需要 AutoDL 已有工程和此前 V1.2 Faithfulness 结果：

- `evaluation/eval_early9_p0.py`；
- `evaluation/early9_pair_genimage_common_v1.py`；
- `evaluation/eval_early9_matched_pair_genimage.py`；
- `evaluation/early9_faithfulness_v1_2/run_early9_rgb_faithfulness.py`；
- `evaluation/early9_rgb_faithfulness_240_v1_2/summary.json`；
- `evaluation/early9_RBCE_vs_RDebias_genimage8000_v1/{summary.json,paired_outputs.npz}`；
- GenImage Eval8000 原图与清单；
- 两组 Step 2625 checkpoint。

所有来源身份及校验 SHA 均会被验证，原图 clean logit 复核。Fake Top-10 如历史 V1.2，仅允许明确核实第十/十一名近并列且分数间隔小于 0.0005 的最多 2 例；统一用**冻结缓存 Top-10**分析，避免因当前 GPU batch 使边界位置轻微交换。例外与 Real 的 Top-10 匹配数会记入元数据。

## 1. 解压、运行本地 CPU 单元测试

上传 `early9_detail4_v1.zip` 至 AutoDL `/root/autodl-tmp/EviRoute_edge/`，执行：

```bash
cd /root/autodl-tmp/EviRoute_edge
conda activate FGTS
unzip -o early9_detail4_v1.zip -d evaluation/
python evaluation/early9_detail4_v1/test_detail4.py
```

预期 `Ran 9 tests ... OK`。测试不加载实际训练权重，包含原版原始视觉统计对照和模拟缓存完整统计流程。

## 2. 冒烟测试（16 Fake + 16 unique Real，两个模型）

```bash
cd /root/autodl-tmp/EviRoute_edge
set -o pipefail
python -u evaluation/early9_detail4_v1/run_early9_detail4.py \
  --root /root/autodl-tmp/EviRoute_edge \
  --out evaluation/early9_detail4_pilot_v1 \
  --stage all \
  --fake-per-source 2 --real-count 16 \
  --batch-size 32 --workers 2 \
  --bootstrap 200 --seed 20261009 \
  2>&1 | tee evaluation/early9_detail4_v1/pilot.log
```

两个模型原图审计以及 RGB metrics 统计完全通过时出现：

`PATCH SCORES EXTRACTION: PASS`
`DETAIL4 CHARACTERIZATION: PASS`

注意：冒烟样本数很小，不要根据冒烟结果选模型。若任一模型原图审计失败，直接停止，不得放宽参数。

## 3. 正式配对分析（240 Fake + 240 **physically unique** Real）

与之前 Faithfulness 240 **完全相同的 240 个 Fake global_index**，每源 30 张；新增 240 张真实图仅在八源共用的 500 张中随机不放回采样一次（共享真实图没有八种生成源属性）。

```bash
cd /root/autodl-tmp/EviRoute_edge
set -o pipefail
python -u evaluation/early9_detail4_v1/run_early9_detail4.py \
  --root /root/autodl-tmp/EviRoute_edge \
  --out evaluation/early9_detail4_240fake_240real_v1 \
  --stage all \
  --fake-per-source 30 --real-count 240 \
  --batch-size 32 --workers 2 \
  --bootstrap 2000 --seed 20261009 \
  2>&1 | tee evaluation/early9_detail4_v1/full.log
```

全程将调用 GPU 进行两个模型共计 960 张图像-模型对的**原图**推理，获取全部 196 个 patch scores；后续 RGB 视觉统计与 Bootstrap 在 CPU 上完成。既不需要为 Detail4 做扰动，也不需要训练。GPU 阶段一旦成功就缓存 `patch_scores.npz` 和 `patch_scores_meta.json`，以后可以重算分析阶段而不进行 GPU 推理。

若 GPU 提取成功、CPU 分析因非模型原因失败，则在不修改源码或原始结果的前提下可重新运行分析：

```bash
python -u evaluation/early9_detail4_v1/run_early9_detail4.py \
  --root /root/autodl-tmp/EviRoute_edge \
  --out evaluation/early9_detail4_240fake_240real_v1 \
  --stage analyze \
  --bootstrap 2000 --seed 20261009
```

若已有 `summary.json` 程序不会覆盖；若进行不同分析设定，应使用独立输出目录，保护实验历史。

## 输出

`evaluation/early9_detail4_240fake_240real_v1/` 包含：

- `summary.json`：未见七源、八源 Fake、唯一 Real 的四项和原版五项 contrast、配对 CI、逐源描述；
- `per_image_detail4.csv`：每张图、每个模型的红蓝位置、四项/五项原始差值和百分位差；
- `per_source_detail4.csv`：各生成源的 Fake 描述性统计；
- `selection_manifest.csv`：逐图 global_index 和实际原图位置；
- `patch_scores.npz`：两个模型的全 196 Patch Score（避免反复推理）；
- `patch_scores_meta.json`：证据来源、身份哈希和原图数值审计。

解读时应优先看 `summary.json -> groups -> fake_unseen7 / real_unique -> paired_ci95 -> detail4_percentile_delta` 和 `per_source_detail4.csv`。负值或正值的大小不应被自动当作模型“证据准确率”。Detail4 是属性描述，不是外部伪影定位真值。原版 7B 的行为与现在 Early-9 的结果应该并排描述而不能假定保持一致。
