# Z世代客群分析（精简代码版）

当前仓库已清理为**仅保留核心代码与必要数据文件**，用于快速展示算法实现与复现实验流程。

## 项目内容

- `algorithm/`：主算法流水线（数据预处理、特征工程、规则分群、XGBoost 分类）
- `DQN/`：不平衡分类的 DQN 实验代码
- `data/`：数据文件（当前包含清洗后数据）
- `run_all.py`：主流程一键执行脚本
- `requirements.txt`：Python 依赖

## 当前目录结构

```text
Z时代数据/
├─ algorithm/
│  ├─ scripts/
│  │  ├─ 01_data_preprocessing.py
│  │  ├─ 02_feature_engineering.py
│  │  ├─ 03_rule_based_segmentation.py
│  │  └─ 04_xgboost_classification_focal.py
│  └─ INDEX.md
├─ DQN/
│  ├─ dqn_imbalanced_classification.py
│  └─ README.md
├─ data/
│  ├─ cleaned_data.csv
│  └─ cleaned_data_with_labels.csv
├─ run_all.py
├─ requirements.txt
└─ README.md
```

## 快速开始

### 1) 环境准备

- Python 3.10+

```bash
pip install -r requirements.txt
```

### 2) 运行主流程

```bash
python run_all.py
```

`run_all.py` 默认执行：
1. `algorithm/scripts/01_data_preprocessing.py`
2. `algorithm/scripts/02_feature_engineering.py`
3. `algorithm/scripts/03_rule_based_segmentation.py`
4. `algorithm/scripts/04_xgboost_classification_focal.py`

## 数据说明

- `01_data_preprocessing.py` 默认读取 `data/data_fixed.csv` 作为原始输入。
- 如果你当前仓库只有 `cleaned_data.csv`，可从第 2 步或第 3 步开始运行脚本（按需执行）。

## 模块导航

- 算法模块快速索引：`algorithm/INDEX.md`
- DQN 说明：`DQN/README.md`

## 说明

- 本仓库为代码精简版，聚焦算法实现与复现。
- 若用于公开展示，请确保数据已完成脱敏与合规检查。
