# Early-9 RGB Faithfulness V1.2（审计边界并列版）

这个版本是 V1.1 的**数值一致性审计修补**，不改变训练权重、测试图像选择、RGB 16x16 patch 自身均值替换、B1/B2-only/interior 三环严格匹配随机对照、原图预处理或统计定义。

## 为什么修改

V1.1 在 240 张 Fake 原图的 R-BCE 预检查中发现：logit 超容差=0；2 张 Top-10 次序改变；1 张 Top-10 成员改变；Top-1 改变=0。其中 `global_index=7952` 的 Top-10 最后一项在缓存的 87 与新推理的 79 之间交换。**真实模型中的两块 score 差距尚未测定**，因此不能提前断言一定是数值并列。这一版在 AutoDL 现场计算分数差距并执行明确的入选门槛。

上一轮严格配对的 R-BCE 推理默认 `batch-size=32`、在 4500 张去重原图上运行，faithfulness 子集的批次排布与之不完全相同；微小浮点计算差异有可能影响第十名附近排序。

## 关键安全门槛（自动运行）

- 已有 Step2625 checkpoint、数据清单、缓存文件 SHA256 核验保持不变。
- 全部原图 logit 差保持 V1.1 容差：`|delta z| <= 0.002 + 0.0001*|z_reference|`。
- Top-10 成员变化时只允许：**原前九名成员集合完全相同，只有第十个名额 1 换 1，Top-1 不变，新算的两个 patch-score 差距小于 `--max-boundary-score-gap`，新结果中原候选图像块按当前分数排序仍在前 12 名内，且例外数不超过 `--max-near-ties`。** 任何违反条件都失败。
- 每次实验始终使用已冻结 `paired_outputs.npz` 中的 Top-10 作为扰动区域（不是新推理 Top-10）；两个模型分别使用各自冻结的 Top-10。
- 全部允许例外均记录 `summary.json -> clean_output_checks -> R_BCE/R_Debias -> allowed_near_tie_details`；另外自动给出 `summary.json -> boundary_tie_sensitivity`，同时排除**两组模型任一组**出现例外的图像，再做同图配对的独立敏感性分析。
- 仅在两组模型完整结束且审计通过后写入最终输出目录。原先失败的 V1.0/V1.1 文件无需删除或覆盖。

## 0. 上传、解压、测试

把 `early9_rgb_faithfulness_v1_2.zip` 上传至 `/root/autodl-tmp/EviRoute_edge/`，再执行：

```bash
cd /root/autodl-tmp/EviRoute_edge
conda activate FGTS
unzip -o early9_rgb_faithfulness_v1_2.zip -d evaluation/
python evaluation/early9_faithfulness_v1_2/test_faithfulness.py
```

预期 `Ran 9 tests ... OK`。

## 1. 只运行 240 张 × 两个模型的原图核验（不扰动）

```bash
cd /root/autodl-tmp/EviRoute_edge
set -o pipefail
python -u evaluation/early9_faithfulness_v1_2/run_early9_rgb_faithfulness.py \
  --root /root/autodl-tmp/EviRoute_edge \
  --samples-per-source 30 --num-random 20 \
  --batch-size 32 --workers 2 --seed 20261009 \
  --max-near-ties 2 --max-boundary-score-gap 0.0005 \
  --preflight-only \
  2>&1 | tee evaluation/early9_faithfulness_v1_2/preflight_240.log
```

观察结果：
- `BOTH ARMS CLEAN PREFLIGHT: PASS`：可以进行下一步。
- `ORIGINAL NUMERICAL AUDIT FAIL`：**不要**增大容差/绕过审计；把 `current_score_gap` 和 `first_mismatches` 贴给助手。原图对齐还没有验证。

## 2. 240 Fake × 20 随机对照 × 2 模型的 RGB 忠实性测试

仅在上一步 PASS 后运行：

```bash
cd /root/autodl-tmp/EviRoute_edge
set -o pipefail
python -u evaluation/early9_faithfulness_v1_2/run_early9_rgb_faithfulness.py \
  --root /root/autodl-tmp/EviRoute_edge \
  --out evaluation/early9_rgb_faithfulness_240_v1_2 \
  --samples-per-source 30 --num-random 20 \
  --batch-size 32 --workers 2 --chunk-size 8 \
  --bootstrap 2000 --seed 20261009 \
  --max-near-ties 2 --max-boundary-score-gap 0.0005 \
  2>&1 | tee evaluation/early9_faithfulness_v1_2/full_240.log
```

输出文件：`summary.json`, `per_image_results.csv`, `selection_manifest.csv`。最终状态 `RGB FAITHFULNESS AUDIT: PASS`。

## 3. 查看诊断和主要结果

```bash
python - <<'PY'
import json
p='evaluation/early9_rgb_faithfulness_240_v1_2/summary.json'
s=json.load(open(p))
for arm in ('R_BCE','R_Debias'):
    a=s['clean_output_checks'][arm]
    print(arm,'clean/n=',a['n'],'changed membership=',a['top10_membership_changed_global_indices'])
    print('near-tie details:',a['allowed_near_tie_details'])
    print('Unseen-7:',s['results'][arm]['unseen7'])
    print('Excluding near-ties:',s['boundary_tie_sensitivity']['results'][arm]['unseen7'])
print('excluded indices:',s['boundary_tie_sensitivity']['excluded_global_indices'])
for k,v in s['bootstrap_ci95']['estimates'].items():
    if k.startswith('unseen7_'): print(k,v)
PY
```

## 统计注意

这是旧 GenImage Eval8000 的探索性分析（非独立未见测试）。原图数值审计成功并不等于生成图像的伪影定位真值验证。Win Rate 使用图像作为样本单位、其 20 次随机对照的平均差值来定义；随机对照次数不是独立图片数。两模型 logit 尺度不能直接用绝对值进行公平排序，跨模型差异仅作描述性分析。
