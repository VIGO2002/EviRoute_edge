# Early-9 双模型 480 图像 RK3588 数值一致性评估 V1

**实验目标**：比较固定 Step 2625 的 R-BCE 与 R-Debias 两组模型在 AutoDL PyTorch 和 RK3588 三核 NPU 上的 **逐图判别分数、Top-10 证据索引、Patch Score 数值误差、Macro-AUC 差异**。

**不会**重新训练、修改权重或重新转换 RKNN；**不会**覆盖原 P0 或已有 16 张 Golden 测试。**这是先前反复使用的 Eval8000 探索性子集，不是独立最终测试集。**

### 文件与前提

工具包 ZIP 内以 `scripts/early9_pair_480_tools_v1/` 为根，包含：

- `prepare_autodl.py`：AutoDL CPU 上复用 `evaluation/early9_detail4_240fake_240real_v1/patch_scores.npz` 和固定的 480 张图片清单，导出 RGB uint8 与两组 PyTorch 196 维参考评分。
- `run_rk3588.py`：开发板上调用真正的 RKNNLite（不是模拟器），获取逐图 196 维 NPU 输出；`--limit 16` 冒烟或 `--limit 480` 全量。
- `analyze_autodl.py`：把开发板的两份输出拷回 AutoDL，CPU 统计预测保持率、Top-10 证据重叠、源级 AUC、Bootstrap CI 和分数误差。
- `common.py`：输入转换、SHA256 身份核对与 Top-10 计算。
- `test_protocol.py`：不依赖实际模型的 CPU 测试。

**使用前必须已有**：

1. AutoDL `/root/autodl-tmp/EviRoute_edge/evaluation/early9_detail4_240fake_240real_v1/` 中的 `patch_scores.npz`、`patch_scores_meta.json`、`selection_manifest.csv`；原始 Eval8000 manifest、paired_outputs.npz 和全部 480 个原始图像文件。
2. RK3588 `/root/EviRoute_RK3588/models/early9_rbce_fp16/` 和 `models/early9_rdebias_fp16/` 中各一个早前成功测试过的 `.rknn` 文件。两份 RKNN 的 SHA256 已写死在代码中：R-BCE `808cecbf...`、R-Debias `40db8e07...`，哈希不符将停止。
3. 开发板 `python3` 包含 `numpy`、`rknnlite`；AutoDL `FGTS` 包含 `numpy`、`Pillow`、`scikit-learn`。

### 0. AutoDL：安装与本地测试

将 `early9_pair_480_tools_v1.zip` 上传到 `/root/autodl-tmp/EviRoute_edge/`：

```bash
cd /root/autodl-tmp/EviRoute_edge
conda activate FGTS
unzip -o early9_pair_480_tools_v1.zip -d .
python -m unittest discover -s scripts/early9_pair_480_tools_v1 -p 'test_protocol.py' -v
```

### 1. AutoDL：制作 480 样本数据包（CPU，无需 GPU）

```bash
cd /root/autodl-tmp/EviRoute_edge
python -u scripts/early9_pair_480_tools_v1/prepare_autodl.py \
  --root /root/autodl-tmp/EviRoute_edge \
  --out /root/autodl-tmp/EviRoute_edge/evaluation/early9_pair_480_v1 \
  2>&1 | tee scripts/early9_pair_480_tools_v1/prepare.log
```

出现 `PREPARE EVAL480: PASS` 后，生成：

- `evaluation/early9_pair_480_v1/`：原始 JSON/NPZ 文件，留在 AutoDL 供后续分析。
- **`evaluation/early9_pair_480_v1.tar.gz`**：需传给开发板的数据包。压缩包内部结构是 `early9_pair_480_v1/manifest.json`、`references.npz`、`inputs_224_uint8.npz`、`transport_sha256.json`。

`inputs_224_uint8.npz` 一次保存 480 张 resize 后的 RGB uint8；开发板一次解压进内存后，每张通过 `float32 / 255 -> CHW -> external CPU patchify [1,768,1,196]` 还原，完全符合训练和以前部署的预处理语义。**不要把 PNG/JPEG 原图重新处理成归一化的 ImageNet 张量。**

### 2. Windows 中转 → RK3588：上传两份文件

从 AutoDL 下载以下两个文件到 Windows，再经 MobaXterm SFTP 上传到**开发板** `/root/EviRoute_RK3588/`：

- `early9_pair_480_tools_v1.zip`（程序）
- `early9_pair_480_v1.tar.gz`（样本）

在 RK3588 上执行：

```bash
cd /root/EviRoute_RK3588
mkdir -p scripts evaluation results
unzip -o early9_pair_480_tools_v1.zip -d .
tar -xzf early9_pair_480_v1.tar.gz -C evaluation/
ls -lh evaluation/early9_pair_480_v1/*.npz
find models/early9_rbce_fp16 models/early9_rdebias_fp16 -maxdepth 1 -name '*.rknn' -type f -ls
```

**注意**：板端无需运行 unittest（测试依赖 `sklearn`，通常只在 AutoDL 的 FGTS 环境安装）；`run_rk3588.py` 本身仅需要 `numpy` 与 `rknnlite`。

### 3. RK3588：每组先做 16 图烟测

在 RK3588 上运行：

```bash
cd /root/EviRoute_RK3588
RBCE_MODEL=$(find models/early9_rbce_fp16 -maxdepth 1 -type f -name '*.rknn' -print -quit)
RDEBIAS_MODEL=$(find models/early9_rdebias_fp16 -maxdepth 1 -type f -name '*.rknn' -print -quit)
test -n "$RBCE_MODEL" && test -n "$RDEBIAS_MODEL" || { echo 'RKNN model missing'; exit 1; }

python3 -u scripts/early9_pair_480_tools_v1/run_rk3588.py \
  --arm R_BCE --model "$RBCE_MODEL" \
  --data-dir evaluation/early9_pair_480_v1 \
  --out-dir results/early9_pair_480_v1 \
  --core 0_1_2 --warmup 10 --limit 16

python3 -u scripts/early9_pair_480_tools_v1/run_rk3588.py \
  --arm R_Debias --model "$RDEBIAS_MODEL" \
  --data-dir evaluation/early9_pair_480_v1 \
  --out-dir results/early9_pair_480_v1 \
  --core 0_1_2 --warmup 10 --limit 16
```

16 张由八源每源选 1 Real + 1 Fake，两个模型使用相同 16 张；会输出 `*_smoke16_board_report.json` 和 `*_smoke16_board_outputs.npz`。**若任何模型的 SHA256/推理接口不正确，停止并把报错发回对话，不要修改校验规则。**

### 4. RK3588：两组各运行完整 480 张

烟测正常后（**同样的模型和程序**）执行：

```bash
cd /root/EviRoute_RK3588
RBCE_MODEL=$(find models/early9_rbce_fp16 -maxdepth 1 -type f -name '*.rknn' -print -quit)
RDEBIAS_MODEL=$(find models/early9_rdebias_fp16 -maxdepth 1 -type f -name '*.rknn' -print -quit)

python3 -u scripts/early9_pair_480_tools_v1/run_rk3588.py \
  --arm R_BCE --model "$RBCE_MODEL" \
  --data-dir evaluation/early9_pair_480_v1 \
  --out-dir results/early9_pair_480_v1 \
  --core 0_1_2 --warmup 10 --limit 480 \
  2>&1 | tee results/early9_pair_480_rbce.log

python3 -u scripts/early9_pair_480_tools_v1/run_rk3588.py \
  --arm R_Debias --model "$RDEBIAS_MODEL" \
  --data-dir evaluation/early9_pair_480_v1 \
  --out-dir results/early9_pair_480_v1 \
  --core 0_1_2 --warmup 10 --limit 480 \
  2>&1 | tee results/early9_pair_480_rdebias.log
```

预计每组 480 次 NPU 调用约 1 分钟上下，以真机为准。本测试计时包含输入字节→float32 还原、CPU Patchify 和 RKNN API 调用，**不属于完整 JPEG/PNG 解码、UI、磁盘 I/O 的端到端 FPS**；更可靠的 FPS 依然使用之前每组三轮 500 次的持续测试。

### 5. 把板端结果传回 AutoDL 做最终离线统计

开发板 `results/early9_pair_480_v1/` 会生成**四个文件**：

- `R_BCE_full480_board_outputs.npz`
- `R_BCE_full480_board_report.json`
- `R_Debias_full480_board_outputs.npz`
- `R_Debias_full480_board_report.json`

使用 Windows/MobaXterm 中转，上传这四个文件至 AutoDL：

`/root/autodl-tmp/EviRoute_edge/evaluation/early9_pair_480_board_results_v1/`

AutoDL 中运行：

```bash
cd /root/autodl-tmp/EviRoute_edge
conda activate FGTS
python -u scripts/early9_pair_480_tools_v1/analyze_autodl.py \
  --data-dir evaluation/early9_pair_480_v1 \
  --board-results-dir evaluation/early9_pair_480_board_results_v1 \
  --out evaluation/early9_pair_480_analysis_v1 \
  --bootstrap 1000 --seed 20261010
```

完成后 `evaluation/early9_pair_480_analysis_v1/` 包含：

- `summary.json`：真实 RK3588 和 PyTorch 的预测一致率、480 张图像的 Top-10 保留分布、平均 Cosine、Logit 误差、逐生成源 AUC、95% CI。
- `per_image_comparison.csv`：每张每模型的决策、证据漂移和数值误差。
- `per_source_comparison.csv`：平衡样本源级指标。

**关于 AUC 口径：**

- `pooled_480` 是 240 Real + 240 Fake（每个真实图像只有一次记录）。
- `per_source_balanced_60` 是每个生成源独立分配 30 Real + 30 Fake，源级 AUC 主用于**硬件前后差异**，而不是与此前 500 Real + 500 Fake 的 Eval8000 AUC 数字绝对值比较。
- `per_source_shared_real_270` 是 30 Fake + 同一批 240 Real（各源共同引用这 240 个原始真实分数）。对其 Macro-AUC 的配对 Bootstrap 会对共享的 240 张 Real **在同一次重采样中只抽样一次**，避免把重复比较当成独立证据。该 AUC 仍不等同于之前的 500 Real + 500 Fake 的 85.62%/85.17%。
- 没有任何一组结果构成独立最终 held-out 泛化结论。

### 故障时不要执行的操作

不要为了完成测试而修改原 RKNN 权重、禁用哈希、修改 normalization、用 numpy f16 输入替换 uint8→float32、提高失败样本的数值容差，或把失败样本从主报告中直接删掉。若已有同名输出目录，请先检查旧结果，再指定一个**新的** `--out-dir` / `--out` 路径；脚本会拒绝静默覆盖。
