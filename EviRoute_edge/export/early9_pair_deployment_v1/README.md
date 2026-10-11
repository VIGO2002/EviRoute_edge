# Early-9 R-BCE / R-Debias paired deployment V1

目标：复用 AutoDL 项目原有 **P0 Early-9 External Patchify FP16** 成功链路，为两份冻结的 Step2625 权重建立**隔离**导出目录。没有训练或模型选优功能。保留旧 P0 完全不覆盖。该工具包中没有权重、ONNX 或 RKNN 成品；需要在 AutoDL 运行之后才会产生。CPU 单元测试不是 AutoDL 或 RK3588 数值验证。

## 源码与依赖核验

工具包基于你提供的 11 个真实源码文件，原有 `export/export_eviroute_early9_p0_onnx.py`、`export/experiment_early9_external_patchify_onnx.py`、`evaluation/eval_early9_p0.py` 及 `scripts/benchmark_early9_external_patchify_fast_board.py`。**两个传递依赖没有出现在本次 tar 中**，但历史 P0 导出器会用到它们：`export/validate_early7_deployment.py` 和 `models/eviroute.py`。新版 `--phase preflight` 会检查服务器本地是否存在，缺少就停止，不能假定已安装。

冻结 SHA256（新流程自动校验）：
- R_BCE Step2625: `3258e32522147a5bb434315e2cb4070f5e348dac65f44b26521f41d94933ebe2`
- R_Debias Step2625: `ddfe015a113d137843427a2f547216b7ee1d84eb3cae28717269b4789e1c0a90`
- strict-pair NPZ: `5bc635bd70b4f8d47ca82bed686639a36857272079bbc769467d3fa8c773d9da`
- Eval8000 manifest: `4d3f05aed0a8d94b4bfe79ddc10e1fc34636e2f18df19f58ba6566b8d5c81a3e`
- original loader: `c04f805b1315b04bd81aa5c199eea3e9435a3a7bf88da0dbd2329aa12ac2c07f`

以上哈希来自上一轮上传的评估结果及对话。如果 AutoDL 本地文件哈希不同，**不要覆盖源文件或放宽判断**，先报告。

## 第 0 步：上传并解压到正确项目目录

```
cd /root/autodl-tmp/EviRoute_edge
unzip -o early9_pair_deployment_v1.zip -d .
conda activate FGTS
python -m unittest export.early9_pair_deployment_v1.test_pair_deployment -v
```

不会覆盖原有 P0 文件：本包只添加 `export/early9_pair_deployment_v1/` 和 `scripts/benchmark_early9_pair_board.py`，说明书位于 `export/early9_pair_deployment_v1/README.md`。

## 第 1 步：仅检查冻结数据、原始部署代码与依赖

```
for arm in R_BCE R_Debias; do
  python -u export/early9_pair_deployment_v1/export_pair.py --arm "$arm" --phase preflight || break
done
```

两组都必须显示 `PAIR DEPLOYMENT PREFLIGHT: PASS`。这一步不会加载模型，也不会调用 GPU。

## 第 2 步：LoRA 合并及 External Patchify 等价性（PyTorch）

```
python -u export/early9_pair_deployment_v1/export_pair.py --arm R_BCE --phase verify 2>&1 | tee export/early9_pair_R_BCE_verify.log
python -u export/early9_pair_deployment_v1/export_pair.py --arm R_Debias --phase verify 2>&1 | tee export/early9_pair_R_Debias_verify.log
```

每个模型使用独立加载器和已校验 checkpoint；16 张 Golden RGB（8 张共享真实样本，仅取一次 + 8 源各 1 张 Fake），分别比对：
- 独立恢复的 LoRA 模型 vs 冻结 strict-pair logits/Top-10，允许最多 2 个经过验证的第 10 名近并列，Logit 容差与早期忠实性实验相同；
- 原始 LoRA 模型 vs 合并的 9-block 模型，每个 Patch Score 参与严格数值比较；
- 原始 RGB PatchEmbed vs CPU 外部 Patchify + 1×1 Conv 投影；
- 外部 Patchify 后输出 vs 原始 LoRA 模型全部 196 个 Patch Scores。

应分别显示 `PAIR LoRA MERGE AUDIT: PASS` 与 `PAIR EXTERNAL PATCHIFY PYTORCH: PASS`。若运行失败，请只发错误输出，**不要继续 ONNX/RKNN**。

说明：用于原始独立 LoRA 恢复的 `evaluation.eval_early9_p0.P0_CKPT` 在 Python 进程内临时指向 R-BCE/R-Debias，随后恢复，**不会修改老源码或保存原 P0 文件**。合并器仍用原 `EviRouteEarly9P0Deploy` 对比独立加载权重。因为老类名称含 P0，并不意味着使用旧 P0 权重。

## 第 3 步：导出两份 ONNX（FGTS 环境）

```
python -u export/early9_pair_deployment_v1/export_pair.py --arm R_BCE --phase export 2>&1 | tee export/early9_pair_R_BCE_export.log
python -u export/early9_pair_deployment_v1/export_pair.py --arm R_Debias --phase export 2>&1 | tee export/early9_pair_R_Debias_export.log
```

成功时都显示 `PAIR EXTERNAL PATCHIFY ONNX: PASS`。生成目录：

```
export/early9_pair_deployment_v1/R_BCE/
  early9_external_patchify_224.onnx
  export_metadata.json
  case_manifest.json
  merge_check.json
  onnx_check.json
  case_00_packed_input.npy ...
  board_cases/
    case_00_input.npy
    case_00_pytorch_scores.npy
    ...
export/early9_pair_deployment_v1/R_Debias/
  (相同结构，文件身份互不混淆)
```

输入：float32 `(1,768,1,196)` CPU 无损 Patchify 结果；输出：`(1,196)` Fake-support scores。图像级 Logit 必须 CPU 计算 Top-10 mean。不能对原 RGB 直接 NPU 输入，也不能在导出环节删除 RoPE、prefix、norm 等步骤。

## 第 4 步：转换 RKNN FP16（rknn232 环境；此时再做）

```
conda activate rknn232
python -u export/early9_pair_deployment_v1/convert_pair_fp16.py --arm R_BCE --sim-cases 8 2>&1 | tee export/early9_pair_R_BCE_rknn.log
python -u export/early9_pair_deployment_v1/convert_pair_fp16.py --arm R_Debias --sim-cases 8 2>&1 | tee export/early9_pair_R_Debias_rknn.log
```

内部明确 `target_platform='rk3588'`、`build(do_quantization=False)`；不得切换到此前证据不稳定的 INT8。两组各自要显示 `RKNN SIMULATOR: PASS`；FP16 RKNN 保存在各自的 `rk3588_fp16/` 目录，文件名不同。`conversion_validation.json` 记录 RKNN SHA256 供板端核对。**模拟器通过不等于真实开发板通过**。

## 第 5 步：真机顺序部署（不要覆盖 P0）

在 PC/AutoDL 将每份 `rk3588_fp16/*.rknn`、对应 `board_cases/` 与本包提供的 `scripts/benchmark_early9_pair_board.py` 拷贝至 RK3588。例如：

```
/root/EviRoute_RK3588/models/early9_rbce_fp16/
  early9_r_bce_external_patchify_fp16.rknn
  board_cases/
/root/EviRoute_RK3588/models/early9_rdebias_fp16/
  early9_r_debias_external_patchify_fp16.rknn
  board_cases/
/root/EviRoute_RK3588/scripts/benchmark_early9_pair_board.py
```

板端分别运行（假设文件已上传）：

```
cd /root/EviRoute_RK3588
python3 scripts/benchmark_early9_pair_board.py \
  --model models/early9_rbce_fp16/early9_r_bce_external_patchify_fp16.rknn \
  --cases-dir models/early9_rbce_fp16/board_cases \
  --core 0_1_2 --warmup 20 --runs 500 \
  --out results/early9_rbce_fp16_board_500.json

python3 scripts/benchmark_early9_pair_board.py \
  --model models/early9_rdebias_fp16/early9_r_debias_external_patchify_fp16.rknn \
  --cases-dir models/early9_rdebias_fp16/board_cases \
  --core 0_1_2 --warmup 20 --runs 500 \
  --out results/early9_rdebias_fp16_board_500.json
```

为防止模型文件错传，优先附加 `--expect-sha256 <conversion_validation.json 中的 rk3588_model_sha256>`。上述脚本来自已经使用过的真机测试脚本，保留 NPU 0/1/2 核、实测对齐、吞吐、延迟、温度和内存指标。

注意：这个板端性能测试只覆盖**预加载 RGB tensor → CPU Patchify → NPU → CPU Top-10**，不含解码/双三次缩放/完整应用/可视化。后续需要再做端到端实拍图像及更大样本的 AUC、Top-10 稳定性测试。

## 故障/限制

- 若 `--phase preflight` 提示缺 `export/validate_early7_deployment.py` 或 `models/eviroute.py`，请先给出缺失路径，不要用这份压缩包替代原始代码。
- 双模型导出的首次模型构建可能耗费 CPU 内存和数分钟；16 张逐图原始模型与合并模型对比也需要一定时间。
- 成功通过本地 unit tests **不代表**原始 P0 权重与训练期权重完全同一，实际以严格冻结 SHA256 为准。
- 当前 GenImage Eval8000 用于迭代分析、不是独立最终测试集。
- 输入 packed tensor 需要 float32；开发板历史 P0 真机 FP16 数值误差和 9.985 FPS 不能被预先宣称为这两个新模型的结果。
