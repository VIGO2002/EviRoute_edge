# Early-9 RK3588 Full JPEG/PNG Detection Pipeline V1

**2026-10-10 · 阶段：完整图像应用首次真机验收，非既有 480 张 NPU/FP16 验证的重新计算。**

## 已验证基础与交付范围

冻结模型身份（文件哈希，不是训练 checkpoint 哈希）：

| Arm | `.rknn` SHA256 |
|---|---|
| R_BCE | `808cecbf53a60d3071d622a32792f7cce782a22115c97b2ead9d7cbda75cf597` |
| R_Debias | `40db8e07ef6fe537c913d19bdbbb0fe01ec0c1f3a1fba14d1e86a6ae6e4a251c` |

**两份模型已经在 RK3588 跑通**，本工具不会修改、转换或覆盖模型；仅创建新 `results/early9_full_pipeline_v1/...`。应用严格保持先前 480 工具 `common.rgb_uint8_to_packed` 的输入协议与 `Top-10 mean(logit)` 决策方式。

**重要证据含义**：无论预测 REAL 还是 FAKE，绘制的均是 *10 个最大假图支持分数位置*，对于 REAL 样本这些不是“真实类局部证据”。Top-10 框不是真值伪造区域标注，sigmoid(logit) 也不是经过置信度校准的概率。

## 0. 安装程序（只需上传这个 ZIP，不必重新上传 RKNN/480 NPZ）

把 `early9_rk3588_image_pipeline_v1.zip` 通过 Windows / MobaXterm 上传到 **开发板** `/root/EviRoute_RK3588/`，然后：

```bash
cd /root/EviRoute_RK3588
unzip -o early9_rk3588_image_pipeline_v1.zip -d .
python3 -c 'import numpy; from PIL import Image; from rknnlite.api import RKNNLite; print("BOARD DEPENDENCIES: PASS")'
ls -lh scripts/early9_full_pipeline_v1/run_images.py
find models/early9_rbce_fp16 models/early9_rdebias_fp16 -maxdepth 1 -type f -name '*.rknn' -ls
```

程序依赖设备的 `numpy`、`Pillow (PIL)` 和 `rknnlite`。如 Pillow 缺失，先报告错误，确认系统当前 apt/pip 环境后再安装；**不应因此修改其他 RKNN 库版本**。

若在 AutoDL 的 FGTS 环境想先查看 CPU 测试（使用模拟 NPU，并非真实 RK3588 验证），将 ZIP 解压到 `/root/autodl-tmp/EviRoute_edge/`：

```bash
python -m unittest discover -s scripts/early9_full_pipeline_v1 -p 'test_*.py' -v
```

## 1. 找到板端已有的 JPEG/PNG 原图

```bash
cd /root/EviRoute_RK3588
find images -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -20
```

若 `images` 不存在或为空，先通过 SFTP 上传 8~20 张代表性原图（建议真实图/生成图都包含）到 `images/full_pipeline_smoke_v1/`。**不需要重新上传 480 NPZ**：`evaluation/early9_pair_480_v1/inputs_224_uint8.npz` 已是缩放张量，用于数值对照；本阶段专门测试真实 JPEG/PNG 的解码、预处理、可视化与端到端 I/O。

## 2. 第一轮：两组各测 8 张原始图、保存对照框图

如原始图放在 `images/full_pipeline_smoke_v1/`，依次运行：

```bash
cd /root/EviRoute_RK3588

python3 -u scripts/early9_full_pipeline_v1/run_images.py \
  --arm R_BCE --input images/full_pipeline_smoke_v1 \
  --out-dir results/early9_full_pipeline_v1/R_BCE_smoke8 \
  --core 0_1_2 --warmup 10 --max-images 8 --save-visual comparison

python3 -u scripts/early9_full_pipeline_v1/run_images.py \
  --arm R_Debias --input images/full_pipeline_smoke_v1 \
  --out-dir results/early9_full_pipeline_v1/R_Debias_smoke8 \
  --core 0_1_2 --warmup 10 --max-images 8 --save-visual comparison
```

两个程序**串行运行**，同一时刻只装载一个模型。若原图位于其他目录，**只更改 `--input` 路径**。目录内文件按相对路径名排序，默认递归查找 JPEG、PNG 等 Pillow 支持的静态文件。模型在原目录下自动找到唯一 `.rknn` 文件，且必须通过前述 SHA256 校验；需要显式路径时使用 `--model path/to/file.rknn`，哈希仍会检查。

成功后，每个结果目录：

```text
results/early9_full_pipeline_v1/R_BCE_smoke8/
├── summary.json           # JPEG解码+可视化+保存的完整处理统计
├── predictions.jsonl      # 逐图输入SHA256、真假、logit、sigmoid、10个Patch编号、逐阶段延迟
└── visuals/
    └── 0000_....png      # 448x224: 左侧224输入图，右侧Top10证据框图
```

报告记录 5 种阶段 `decode_resize_ms`（含输入文件 SHA256）、`cpu_patchify_ms`、`rknn_inference_ms`、`cpu_top10_ms`、`visual_write_ms`，以及 `full_image_ms`/总图片每秒数。排除模型加载、warmup、最后 summary 写入及 GUI/网络；**包含图片磁盘读写和 PNG 保存**，因此与先前预加载张量的 9.5 FPS 不是同一测量口径。

保存绘制框图的坐标以 `224×224` 预处理图为准，不是原始图片自然尺寸。排序最高的 Patch 使用更明显红框；其他九个使用橙色框。图片两侧不加视觉文字，分数和坐标保存在 JSONL。

## 3. 核验对照并查看完整结果

在板端运行：

```bash
python3 scripts/early9_full_pipeline_v1/compare_runs.py \
  --rbce-run results/early9_full_pipeline_v1/R_BCE_smoke8 \
  --rdebias-run results/early9_full_pipeline_v1/R_Debias_smoke8 \
  --out-dir results/early9_full_pipeline_v1/pair_smoke8_audit
```

`compare_runs.py` 要求两组记录的 **原始图片相对路径、排序、输入文件 SHA256、预处理和 NPU 核心设置一致**，输出 `comparison_summary.json` 与 `paired_predictions.csv`。其中 **inter_model_top10_overlap 指模型与模型间的证据差异，不能用来替代各模型 NPU 对 PyTorch 的证据一致性**；后者已经由先前480评估独立测过。

## 4. 第二轮：正式完整 JPEG/PNG 应用测试

冒烟测试通过后，使用 **同一目录至少 30 张以上图片**做完整系统性能测试；样本尽量覆盖不同原始分辨率、JPG/PNG、Real/Fake。建议同一批图片分别运行 R-BCE、R-Debias，每次用新的输出目录：

```bash
python3 -u scripts/early9_full_pipeline_v1/run_images.py \
  --arm R_BCE --input images/full_pipeline_test_v1 \
  --out-dir results/early9_full_pipeline_v1/R_BCE_full \
  --core 0_1_2 --warmup 20 --save-visual comparison

python3 -u scripts/early9_full_pipeline_v1/run_images.py \
  --arm R_Debias --input images/full_pipeline_test_v1 \
  --out-dir results/early9_full_pipeline_v1/R_Debias_full \
  --core 0_1_2 --warmup 20 --save-visual comparison
```

**为了隔离证据可视化写盘成本**，可将 `--save-visual comparison` 改为 `--save-visual none`，改用另一组新输出目录，对照纯图像解码+预处理+NPU+后处理的图像每秒数。两个结果都属于端到端子协议，不能与 480 `.npz` 评估或持久基准偷换口径。

正式对比：

```bash
python3 scripts/early9_full_pipeline_v1/compare_runs.py \
  --rbce-run results/early9_full_pipeline_v1/R_BCE_full \
  --rdebias-run results/early9_full_pipeline_v1/R_Debias_full \
  --out-dir results/early9_full_pipeline_v1/pair_full_audit
```

**如有完整且独立核对过的 Real/Fake 标签**，可用 `--labels-csv labels.csv`。标签 CSV 必须包含精确相对路径与 `label` 列（`0=Real, 1=Fake`），覆盖全部图片且无重复；它会启用 sklearn 计算 AUC/AP/ACC，建议在 AutoDL 的 FGTS 环境运行离线 `compare_runs.py`（开发板不一定有 sklearn）。未提供标签时**不会凭路径名假定真假标签**。

## 5. 这阶段的验收条件

- 两组 RKNN SHA256 与既定模型哈希一致，`load_rknn`/`init_runtime` 成功。
- 输入必须是真正 JPEG/PNG 原图；不存在文件转换错误、误用 ImageNet mean/std、推理输出尺寸错误/非有限值。
- 保存的图片中十个框严格对应报告的 `top10_patch_indices`，224x224 patch grid 14×14；FAKE/REAL 与 Top10-mean 的零阈值一致。
- 两组图片顺序、SHA256、预处理协议、NPU 核心掩码对齐。
- 分别统计完整图像的吞吐量/平均/p95、NPU 单独延迟、CPU Patchify、JPEG/PNG 读写和可视化成本。图像解码/保存路径、文件系统缓存、PNG/JPEG 混合比例会影响结果。
- 温度为内核 thermal zone 读数，并不等于功耗；功耗 <10 W 需要额外可靠测量。
- 本流程不证明新数据域泛化或区域真实性。用于泛化评价需要一个严格独立、标签可信的 holdout（不可重新使用已反复查看的 Eval8000 子集）。

**不要覆盖此前 `/root/EviRoute_RK3588/results/early9_pair_480_v1/`、P0 模型及现有 Golden 文件。** 程序在输出目录已非空时会直接拒绝写入。
