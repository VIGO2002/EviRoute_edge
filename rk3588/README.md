# EviRoute-Edge · RK3588 板端精简核心源码（V1）

> 精简整理日期：2026-10-12。源文件来自已上传的 RK3588 板端归档；原 `.py/.js/.css/.html/.sh` 字节保持不变，保留实际使用的相对目录，**未对推理逻辑重构**。

## 功能范围

- R-BCE / R-Debias：RK3588 三核 NPU，RKNN FP16，224×224 RGB，CPU External Patchify `(1,768,1,196)`，196 patch scores，Top-10 mean Logit。
- **串行基线**：`scripts/early9_full_pipeline_v1/run_images.py`。
- **正式异步流程**：`scripts/early9_async_pipeline_v1/run_async.py`，直接导入上述串行程序中的公共数值/可视化实现，因此两者都需要保留。
- **离线 Web GUI**：`scripts/eviroute_edge_ui_v1/server.py` 及 `static/`，在板端回环地址运行；GUI 通过子进程调用异步推理脚本。
- **工程实验**：Golden 对照、480 张跨设备板端采集、160 张原图端到端基线与异步对照（保留必要的板端执行/复核代码）。

## 运行前的必要条件

这是一份**源码展示和板端工程核心代码包，不是开箱即用的系统镜像**。需要：

1. iTOP-RK3588、Ubuntu 22.04 / aarch64、匹配的 RKNNLite2 运行库与 NPU 驱动。
2. Python 3 + NumPy + Pillow；运行 `server.py` 不依赖外部云服务，Kiosk 显示需要 Chromium 与图形桌面。
3. 两个真实 `.rknn` 权重分别放在 `models/early9_rbce_fp16/` 和 `models/early9_rdebias_fp16/`，且 SHA256 必须与 `run_images.py` 中的固定哈希一致。**权重不包含在本 ZIP 中。**
4. 评估脚本需对应的 Golden `.npy` / Eval480 `.npz` / 160 原图与旧结果；这些大文件也不随源码发布。
5. 原程序默认项目路径 `/root/EviRoute_RK3588/`，部署到其他路径时需按各脚本提供的 `--project-root` 参数或启动脚本实际修改路径；切勿直接覆盖已工作项目。

## 核心调用链

```text
scripts/eviroute_edge_ui_v1/server.py
  └── scripts/early9_async_pipeline_v1/run_async.py
       └── scripts/early9_full_pipeline_v1/run_images.py
            ├── 预处理 / External Patchify
            ├── RKNNLite 输入输出 / Top-10 / 判别
            └── 证据框选 / 用时统计
```

上述依赖在精简包中**均实际存在**。`scripts/eviroute_edge_ui_v1/static/` 的全部三个前端文件亦已包含。`start_server.sh` 在本地 `127.0.0.1:8765` 监听，默认无对外开放；不应在没有认证与网络访问控制的情况下修改为 `0.0.0.0`。

## 已验证的原环境结果（非本 ZIP 重新测得）

- 已报告：480 张跨设备一致性：两个模型分别 475/480 预测一致，平均 Top-10 重叠约 9.33/10。
- 已报告：160 张原始图像 3 轮异步吞吐量：R-BCE 保存证据图约 10.624 FPS，R-Debias 约 10.600 FPS；相对串行加速约 1.508 倍。连续吞吐量≠单图完成延迟。
- 仅执行代码归档、语法与 CPU/mock 测试。**未在这里执行真实 NPU，也不声明发布包包含模型权重和独立数据集。**

## CPU/mock 基础检查

在装有 NumPy 与 Pillow 的环境中，从项目根运行：

```bash
python3 -m unittest discover -s scripts/early9_full_pipeline_v1 -p 'test_*.py' -v
python3 -m unittest discover -s scripts/early9_async_pipeline_v1 -p 'test_*.py' -v
python3 -m unittest discover -s scripts/eviroute_edge_ui_v1 -p 'test_*.py' -v
```

用于已配好驱动、模型与显示器的 RK3588 环境：

```bash
cd /root/EviRoute_RK3588
bash scripts/eviroute_edge_ui_v1/start_server.sh
# 在另一终端：
bash scripts/eviroute_edge_ui_v1/launch_kiosk.sh topeet
```

注意：GUI V1 每个任务仍启动一个推理进程，**尚未实现常驻 NPU 模型、系统服务开机自启、U 盘自动管理或蓝牙收图**。

## 仓库边界与安全

`eviroute_edge_data/` 保存用户上传索引、文件名与历史任务，不应纳入公开源码仓库。`images/`、`results/`、`logs/`、`evaluation/`、原始 P0/Early7/Early8 旧代码也没有包含在这份板端核心包中；如果需要发表实验复现材料，请分开展示经脱敏、可追溯的实验结果。

详见 `docs/SOURCE_AUDIT.md`、`docs/SOURCE_SHA256.txt`。此 ZIP 是仅供检查/整合的独立副本，未修改用户原始板端文件。
