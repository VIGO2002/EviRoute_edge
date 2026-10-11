# EviRoute-Edge｜AutoDL 精简源码审阅包（V1）

这是从用户上传的 `/root/autodl-tmp/EviRoute_edge/` 原始归档**逐文件筛选**出的展示/审查版本。
**不修改任何原始 Python 文件，不删除服务器原始目录。** 原路径保持不变，便于回溯 SHA256 和依赖。

## 保存范围

- `distillation/`：Early-9 LoRA 架构与 R-BCE/R-Debias Step2625 严格配对训练。
- `evaluation/`：GenImage 配对评估、RGB Faithfulness v1.2、Detail4 与 480 样本跨设备分析。
- `export/`：在已验证的历史 Early-9 合并/External Patchify 基础上的双模型 ONNX/RKNN FP16 导出。
- `scripts/`：RK3588 Golden 验证、480 图片协议、160 张原始图像基准脚本。
- `models/`：导出所依赖的原始评分器定义。
- `tests/`：上述主链路相关小规模测试。
- 核心实验 `summary.json`、源级 CSV、转换哈希记录和方法说明。

**保留某些名字中包含 `P0`、`Early7` 或 `legacy` 的文件，是因为最终源码确实 import/复用了它们。** 不能只依据文件名删除。

## 主动排除

- Early-7/Early-8 深度探索、KD/INT8/hybrid 调参、已废弃的多轮 P0/Debias 训练版本。
- 重复的 Faithfulness v1/v1.1、Detail4 pilot、旧的早期输出与大量 history/smoke 文件。
- 图像、压缩包、模型权重、ONNX/RKNN 二进制、逐张缓存、`.npy/.npz`、日志。
- 已有测试集的原始路径清单 CSV 及原始训练 20k 行 CSV：这些文件**不是代码**，留在原服务器即可。导出及复现实验需要时应从原项目恢复。严格 SHA 校验不应绕过。

## 重要边界：不是独立可运行发布包

此包只用于**审阅/展示/结构整理**；直接解压到一台空机器无法完成训练或导出，因为缺少：

1. 预训练 DINOv3 权重、配对的 `final_step.pt`、初始化/base checkpoint；
2. `distillation/data_manifests/sd14_train20k_seed20261008.csv`、`evaluation/genimage_8domain_eval8000/manifest.csv`；
3. Eval8000 `paired_outputs.npz`、Detail4 `patch_scores.npz`、480 `references.npz`、测试原图等；
4. 对应 Python/AutoDL CUDA/RKNN Toolkit2 环境。

这些文件在原服务器上有已验证的路径和哈希，不应通过修改源代码中的 SHA 来规避缺失检查。

## 目前最重要的缺口：开发板软件不在本次上传包中

本次材料只取自 AutoDL 的 `EviRoute_edge`，因此不含后来在 RK3588 上开发的：

- `scripts/early9_full_pipeline_v1/`：JPEG/PNG 原始图片 GUI 前的实际推理流程；
- `scripts/early9_async_pipeline_v1/`：三阶段异步流水线、160 图吞吐量优化及一致性验证；
- `scripts/eviroute_edge_ui_v1/`：设备本地离线 Web UI/后端、U 盘导入、证据显示与报告导出。

等开发板可访问时，应从 `/root/EviRoute_RK3588/` 单独采集上述 **3 个目录的源码**，再做真正的产品级统一工程。

## 阅读顺序

1. `distillation/early9_p0_matched_pair_core.py` → `train_early9_p0_matched_pair.py`；
2. `evaluation/eval_early9_matched_pair_genimage.py` + Faithfulness/Detail4；
3. `export/early9_pair_deployment_v1/export_pair.py` → `convert_pair_fp16.py`；
4. `scripts/early9_pair_480_tools_v1/` → `scripts/early9_e2e_original160_v1/`。

## 严格性

当前 GenImage Eval8000/Detail4/480 样本均是已探索数据，不属于全新独立的最终测试集。现有跨设备指标验证的是同图输出/排序保持，并不直接证明真实伪影定位或板端因果忠实性。

详见 `CODE_FILE_MANIFEST.csv`（逐文件分类和 SHA256）。
