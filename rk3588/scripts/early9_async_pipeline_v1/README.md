# RK3588 Early-9 双模型异步流水线 V1

**目的**：不更改两份已经验证过的 `.rknn` 文件及冻结阈值，在相同 **160 张原始 JPEG/PNG** 上，严格比较串行和三阶段有界异步流水线的吞吐量，并核验输入 SHA256、Logit、预测、**有序** Top-10 一致性。

## 完全复用的已有文件

必须已存在：

- `/root/EviRoute_RK3588/scripts/early9_full_pipeline_v1/run_images.py`（此前已成功运行的串行实现）
- `/root/EviRoute_RK3588/models/early9_rbce_fp16/*.rknn`；`models/early9_rdebias_fp16/*.rknn`
- `/root/EviRoute_RK3588/images/full_pipeline_original160_v1/`（此前 160 原始图片）
- `/root/EviRoute_RK3588/results/early9_full_pipeline_v1/original160_formal3_v1/`（此前串行三轮报告及 JSONL，严格同图对照用）

新的 Python 代码**仅**写入 `scripts/early9_async_pipeline_v1/`，新的实验仅写入 `results/early9_async_pipeline_v1/`。不会修改旧模型、旧脚本、旧结果。

三阶段：

1. Decode 线程：读原始文件、SHA256、Pillow RGB+Bicubic 224、外部 CPU Patchify，有限队列暂存。
2. **主线程独占 RKNNLite**：唯一 NPU 实例、唯一 `.inference` 调用者、CPU Top-10。**不并发运行两组 RKNN**。
3. Writer 线程：证据框选、PNG 写盘及 JSONL 逐图结果，保持输入顺序。

最大队列深度默认 3，防止 RAM 无界增长。保留相同 `top10_patch_indices` 和 `logit` 计算函数；严格检查 `.rknn` SHA256。

## 1. 安装（开发板）

上传 `early9_async_pipeline_v1.zip` 至 `/root/EviRoute_RK3588/`：

```bash
cd /root/EviRoute_RK3588
unzip -o early9_async_pipeline_v1.zip -d .
python3 -m unittest discover -s scripts/early9_async_pipeline_v1 -p 'test_async.py' -v
```

本地 CPU 测试模拟 NPU，验证同一批 JPEG/PNG 的串行与异步预测、Logit、Top-10（顺序也必须一致）、PNG 字节一致；它**不是**真机推理的替代。

## 2. 先测 8 张（真机）

```bash
cd /root/EviRoute_RK3588
python3 -u scripts/early9_async_pipeline_v1/run_async.py \
  --arm R_BCE \
  --input images/full_pipeline_original160_v1 \
  --out-dir results/early9_async_pipeline_v1/rbce_pilot8 \
  --core 0_1_2 --warmup 10 --queue-depth 3 \
  --max-images 8 --save-visual comparison
```

如果输出 `ASYNC PIPELINE: COMPLETE R_BCE`，然后做完整的同图比较；如果测试输出目录已存在，不要覆盖，改用新目录名称。

## 3. 正式 160 张，三轮、双模型、双模式

```bash
cd /root/EviRoute_RK3588
python3 -u scripts/early9_async_pipeline_v1/benchmark_pair.py \
  --tag original160_async_formal3_v1 \
  --rounds 3 --warmup 20 --queue-depth 3
```

每组单独起 Python/RKNN 进程，共 12 组 × 160 张，模型和模式运行先后交错，避免固定次序偏差。预计几分钟（依设备实时情况）。

## 4. 与此前串行三轮逐图严格对照

```bash
cd /root/EviRoute_RK3588
python3 scripts/early9_async_pipeline_v1/audit_vs_serial.py \
  --serial-root results/early9_full_pipeline_v1/original160_formal3_v1 \
  --async-root results/early9_async_pipeline_v1/original160_async_formal3_v1 \
  --rounds 3 --logit-tol 0.00001
```

成功标志：`ASYNC VS SERIAL AUDIT: PASS`。会输出四类 FPS 对照及相对加速比，并写入新目录内 `audit_vs_serial.json` 和 `audit_vs_serial.csv`。如出现预测/Top-10 改变或超过阈值，审计会明确失败；不要为了获得 PASS 而随意放宽阈值。

## 科学口径注意事项

- `full_pipeline_images_per_second` = **整批图片数/实际流水线 wall time**，最适合比较吞吐；`phase_ms` 反映各线程操作耗时，**异步阶段均值不可以相加得到 wall latency**。
- 单张 `full_image_ms` 包括线程/队列等待，可能不会下降；这是正常的吞吐/延迟权衡。
- 读取/解码、NPU、PNG 写盘可能在不同线程重叠，实际加速取决于 Python GIL、CPU/NPU 共享内存带宽、驱动执行及频率调节；不能保证达到 10 FPS。
- 旧测试的 `none` 和 `comparison` NPU 用时存在运行顺序差异，不能解释为写盘加速了 NPU。本脚本每轮轮换模式次序，缩小这一混杂。
- 不测电功率；温度只是 `/sys/class/thermal` 内核热区采样，不能推出 `<10W`。
- 此时无标签泛化能力的新实验；原始 160 张来自之前 480 子集，**不是**全新独立 holdout。
- 所有 Top-10 是最高 Fake-support patch，不能描述成已经验证的“伪造区域真值”。

## 显示屏 / 产品化路径

先让 RK3588 通过 Wi-Fi/LAN 在本地启动 Web UI：选择图片、切换模型、查看真假 Logit 和 Top-10 框图、下载 JSON。局域网电脑/手机可先作为显示与操作终端。随后才给产品外壳加入标准 HDMI 输入 + USB HID 触摸的 7/10 英寸 IPS 屏，复用同一 Web UI。本 V1 优化包**尚未实现 Web UI**；本阶段只验证异步流水线。
