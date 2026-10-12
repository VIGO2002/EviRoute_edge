# RK3588 源码审计与精简记录 · 2026-10-12

## 原始归档核验

- 输入压缩包：`EviRoute_RK3588_source_review.tar.gz`，128 KiB 量级。
- 共 **161 个文件**：`scripts/` 36、`src/` 13、`tests/` 1、`models/` 2、`evaluation/` 9、`eviroute_edge_data/` **98**、项目根 README **2**。
- 后缀：42 `.py`、3 `.sh`、1 `.html`、1 `.css`、1 `.js`、4 `.md`，其余是 JSON/TXT。
- 仅发现 **1 组字节完全相同** 的源码副本：
  `scripts/benchmark_early9_pair_board.py` == `scripts/benchmark_early9_external_patchify_fast_board.py`。
- 多个近似但非完全相同的历史程序：`benchmark_early8_*` / `benchmark_early9_p0_board.py`、`src/run_eviroute_early7*.py` 与 `src/run_eviroute_eval200*.py`；不属于当前正式双模型主链路。

## 必须保留的运行依赖

1. `scripts/eviroute_edge_ui_v1/server.py` + `static/index.html` + `static/style.css` + `static/app.js`；浏览器/GUI。
2. `scripts/early9_async_pipeline_v1/run_async.py`；产品检测主要引擎，主线程独占 RKNNLite，CPU 预处理/PNG 写入异步。
3. `scripts/early9_full_pipeline_v1/run_images.py`；**异步模块会在运行时 `import run_images as old`**，不能作为所谓“重复串行版本”删掉。
4. `models/early9_*/conversion_validation.json`；保留权重来源和哈希审计，但不包含实际 `.rknn`。
5. 重要测试：核心单元测试、串/异步一致性脚本以及 Golden/480/160 的必要板端代码。

## 精简规则

**保留：** 当前双模型串行/异步两条可核对的执行链、GUI 完整前后端、必要的单元测试、最小实验复核和模型 SHA256 配置。

**排除：**

- `eviroute_edge_data/` 中 **98 个**运行时文件（13 组 job 记录/报告及上传索引等）——非源码，可能含原文件名与用户操作记录，不宜上传公共仓库。
- `src/` **13 个**历史 Early2/7/旧 P0 部署或 Eval200 工具，当前产品主链路未直接引用。
- `scripts/` 根下旧 Early8/9-P0 测试、旧 `detect_early9_image.py`、旧 runtime profiler 等不参与当前产品主路径。
- `scripts/early9_pair_480_tools_v1/{prepare_autodl.py,analyze_autodl.py,test_protocol.py,README.md}`：这些属于 AutoDL 端生成/统计或依赖被排除 AutoDL 工具的跨环境测试；板端仅保留 `common.py` 与 `run_rk3588.py`。
- `scripts/early9_e2e_original160_v1/{prepare_original160_autodl.py,test_original160.py}`：AutoDL 数据准备/相关测试应放在训练/数据准备侧，不属于板端运行核心。
- `/evaluation` 下所有小 JSON/TXT 历史结果，因不包含完整实验输出，另行作为结果档案保存更准确。
- 其他旧 README / 临时脚本、二进制模型、图像、缓存。

## 完整性边界

- **当前板端 V1 产品运行链路源码：齐全。**
- **可直接在新 RK3588 机器上独立部署：不齐全。** 缺 `.rknn` 权重、RKNNLite2 二进制/驱动、预设环境、图片样本与相关服务配置。
- **未来 V2 功能：尚未实现**。包括常驻 NPU 模型、U盘自动挂载/安全移除、USB 独立保存位置选择、开机自动启动、蓝牙传图、功耗监测等，不能算作“打包遗漏”。
- `start_server.sh` 和 `launch_kiosk.sh` 写死了 `/root/EviRoute_RK3588` / `topeet`，本次为忠实保留已验证原文件，未做参数化重构。
- `server.py` 默认只监听 127.0.0.1，不提供账户认证，保持回环访问，不可直接对局域网开放。

## 下一步建议

将 AutoDL 训练/模型导出源码与本包组成双端仓库，但暂时**不重写 import 路径**。建立统一的 README、环境矩阵、模型权重校验表，以及使用独立 `results/` 存放实验报告。后续在 RK3588 上完成一轮等价回归后，才能考虑将公共函数提取为 SDK，并逐步替换历史相对路径。
