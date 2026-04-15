# Algorithm 模块索引（30秒版）

## 这个模块做什么

将原始用户行为数据处理为可训练特征，完成规则分群 + XGBoost 多分类，输出模型文件、客群结果和评估图表。

## 快速入口

- 总入口：`../run_all.py`
- 详细说明：`README.md`
- 核心脚本目录：`scripts/`

## 一键执行链路

1. `scripts/01_data_preprocessing.py`（清洗）
2. `scripts/02_feature_engineering.py`（特征）
3. `scripts/03_rule_based_segmentation.py`（规则分群）
4. `scripts/04_xgboost_classification_focal.py`（分类训练）

## 关键产物

- 模型：`models/xgboost_model.pkl`、`models/feature_scaler.pkl`、`models/label_encoder.pkl`
- 数据：`outputs/data/segments_data.json`、`outputs/data/model_metrics.json`
- 图表：`outputs/visualizations/`

## 评审最该看什么

- `outputs/data/model_metrics.json`（效果指标）
- `outputs/data/segments_data.json`（客群结果）
- `outputs/visualizations/classification/`（模型可解释图）
