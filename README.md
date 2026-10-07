# FECA Final - 运行指南

## 环境

```bash
conda activate torch
cd /data/coding/FECA_final
```

Python: `/data/miniconda/envs/torch/bin/python`

---

## 数据集

| 数据集 | 交互数 | 用户 | 物品 | 说明 |
|--------|--------|------|------|------|
| hdf-9580 | 9,580 | 1,856 | 583 | 原始 HDF 数据集 |
| hdf-11000 | 11,000 | 1,856 | 583 | hdf-9580 扩充版 |
| healthcaremagic-mimiciii-12k | 11,505 | 2,027 | 1,856 | MIMIC-III 子集 |
| healthcaremagic-mimiciii | 24,153 | 2,673 | 4,284 | MIMIC-III 完整版 |

---

## 运行单个模型

### FECA-optimized（推荐配置）

```bash
python run_customize_model.py --model FECA \
    --dataset hdf-9580 \
    --config config/feca_pu_optimized.yaml \
    --config config/model_config.yaml
```

各数据集的 FECA-optimized 配置相同（`config/feca_pu_optimized.yaml`），核心参数：
- `pu_score_weight: 0.30`
- `pu_penalty_mode: both`

对于 hdf-9580 原始数据集，使用更轻的惩罚效果更好：
```bash
# hdf-9580 专用（pu_score_weight=0.04, time_only）
python run_customize_model.py --model FECA --dataset hdf-9580 \
    --config config/model_config.yaml \
    --config <(echo -e "seed: 2026\nembedding_size: 128\nn_layers: 2\nreg_weight: 3.0e-06\ntrain_batch_size: 1024\nlearning_rate: 0.001\nalignment_loss_weight: 0.015\nenable_pu_score_adjustment: true\npu_score_weight: 0.04\npu_penalty_mode: 'time_only'")
```

### Baseline 模型

```bash
# 所有 baseline 使用 recbole/properties/model/<MODEL>.yaml 中的默认参数
python run_customize_model.py --model RecVAE --dataset hdf-9580 --config config/model_config.yaml
python run_customize_model.py --model MacridVAE --dataset hdf-9580 --config config/model_config.yaml
python run_customize_model.py --model SimGCL --dataset hdf-9580 --config config/model_config.yaml
python run_customize_model.py --model LightGCN --dataset hdf-9580 --config config/model_config.yaml
python run_customize_model.py --model DiffRec --dataset hdf-9580 --config config/model_config.yaml
python run_customize_model.py --model MultiVAE --dataset hdf-9580 --config config/model_config.yaml
python run_customize_model.py --model NAIS --dataset hdf-9580 --config config/model_config.yaml
python run_customize_model.py --model RaCT --dataset hdf-9580 --config config/model_config.yaml
python run_customize_model.py --model GCMC --dataset hdf-9580 --config config/model_config.yaml
```

### 新增公平性模型

```bash
# MetaFairBPR
python run_customize_model.py --model MetaFairBPR --dataset hdf-9580 \
    --config config/metafairbpr.yaml --config config/model_config.yaml

# CofairDSPT1
python run_customize_model.py --model CofairDSPT1 --dataset hdf-9580 \
    --config config/cofair_dspt1.yaml --config config/model_config.yaml
```

---

## 生成完整报告

一次性跑所有模型并生成 PatientUtility 排名报告：

```bash
# 单个数据集
python run_full_reports.py hdf-9580

# 多个数据集
python run_full_reports.py hdf-9580 hdf-11000

# 全部四个
python run_full_reports.py hdf-9580 hdf-11000 healthcaremagic-mimiciii-12k healthcaremagic-mimiciii
```

报告输出到 `feca_results/<dataset>_seed2026_patientutility.md`。

注意：healthcaremagic-mimiciii 数据集较大，每个模型需要 10-20 分钟，全部跑完约 2-3 小时。建议用 nohup：

```bash
nohup python run_full_reports.py healthcaremagic-mimiciii > /tmp/report.log 2>&1 &
tail -f /tmp/report.log
```

---

## 评估指标

| 指标 | 说明 | 方向 |
|------|------|------|
| Recall@10 | 推荐命中率 | ↑ |
| NDCG@10 | 排序质量 | ↑ |
| PatientUtility@10 | NDCG × (1-NormCost) × (1-NormWaitTime) | ↑ |
| DoctorUtility@10 | FairExposure / MismatchPenalty | ↑ |
| TailPercentage@10 | 长尾物品推荐比例 | ↑ |
| AveragePrice@10 | 推荐物品平均价格 | ↓ |
| AverageTime@10 | 推荐物品平均响应时间 | ↓ |

---

## 当前最佳排名（FECA-optimized vs baselines）

| 数据集 | Recall | NDCG | PU | DU | TailPct |
|--------|:------:|:----:|:--:|:--:|:-------:|
| hdf-9580 | #1 | #1 | #1 | #1 | #6 |
| hdf-11000 | #1 | #1 | #1 | #1 | #3 |
| hcm-mimiciii-12k | #2 | #1 | #1 | #3 | #2 |
| hcm-mimiciii | #4 | #4 | #1 | #1 | #2 |

---

## 文件结构

```
FECA_final/
├── run_customize_model.py      # 主运行入口
├── _run_feca.py                # FECA 快捷运行
├── run_full_reports.py         # 批量报告生成
├── config/
│   ├── model_config.yaml       # 数据集字段和指标配置
│   ├── feca_pu_optimized.yaml  # FECA-optimized 参数
│   ├── metafairbpr.yaml        # MetaFairBPR 参数
│   └── cofair_dspt1.yaml       # CofairDSPT1 参数
├── dataset/
│   ├── hdf-9580/               # .inter .item .user
│   ├── hdf-11000/
│   ├── healthcaremagic-mimiciii-12k/
│   └── healthcaremagic-mimiciii/
├── feca_results/               # 评估报告
│   ├── hdf-9580_seed2026_patientutility.md
│   ├── hdf-11000_seed2026_patientutility.md
│   ├── healthcaremagic-mimiciii-12k_seed2026_patientutility.md
│   └── healthcaremagic-mimiciii_seed2026_patientutility.md
└── recbole/                    # 精简后的 RecBole 框架
    ├── model/general_recommender/  # FECA + 10 baselines
    ├── evaluator/                  # PatientUtility + DoctorUtility
    └── properties/model/           # Baseline 默认参数
```
