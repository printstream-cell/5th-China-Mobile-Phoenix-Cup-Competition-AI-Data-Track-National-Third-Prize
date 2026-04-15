# Z世代客群分析系统（GitHub 发布版）

一个面向 15-30 岁用户的客群分析项目，包含：
- 数据处理与建模流水线（`algorithm/`）
- 不平衡分类的 DQN 研究实现（`DQN/`）
- Python FastAPI 服务（`api/`）
- Java Spring Boot 演示后端与页面（`backend(1)/`）

> 该仓库为“算法 + API + 后端演示”多模块集合仓库，适合展示完整技术链路与实验过程。

## 1. 项目目标

- 基于用户行为数据识别 6 大 Z 世代客群
- 输出可解释的客群画像、指标和可视化结果
- 提供 API 与 Web 后端接入能力

六大客群：
- 剧综重度用户
- 硬核玩家
- 社交达人
- 学习型青年
- 电商剁手党
- 音乐发烧友

## 2. 仓库结构（整理后）

```text
Z时代数据/
├─ algorithm/                 # 主算法流水线（推荐主入口）
│  ├─ scripts/                # 01~04 训练/分群脚本
│  ├─ models/                 # 模型产物（pkl）
│  ├─ outputs/                # 数据、指标、图表输出
│  ├─ README.md               # 算法细节
│  └─ INDEX.md                # 模块快速导航
├─ DQN/                       # DQN 不平衡分类实验
├─ api/                       # FastAPI 接口服务
│  └─ INDEX.md                # 模块快速导航
├─ backend(1)/                # Spring Boot 后端 + 静态页面
│  └─ INDEX.md                # 模块快速导航
├─ docs/
│  └─ archive/                # 过程性中文报告归档
├─ run_all.py                 # 算法一键执行入口
├─ requirements.txt           # 根目录算法依赖
└─ README.md                  # 本文件
```

## 3. 快速开始（推荐）

### 3.1 环境

- Python 3.10+
- 建议在虚拟环境中运行

```bash
pip install -r requirements.txt
```

### 3.2 数据准备

将原始数据放置到：

```text
data/data_fixed.csv
```

### 3.3 一键执行算法流水线

```bash
python run_all.py
```

当前 `run_all.py` 执行顺序为：
1. `01_data_preprocessing.py`
2. `02_feature_engineering.py`
3. `03_rule_based_segmentation.py`
4. `04_xgboost_classification_focal.py`

## 4. 各模块启动方式

### 4.1 FastAPI 服务

```bash
cd api
pip install -r requirements.txt
python app.py
```

文档地址：
- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

### 4.2 Spring Boot 后端（演示）

```bash
cd "backend(1)"
mvn spring-boot:run
```

默认访问：`http://localhost:8080`

## 5. 关键输出

算法模块主要产物：
- `algorithm/models/`：`xgboost_model.pkl`、`feature_scaler.pkl`、`label_encoder.pkl`
- `algorithm/outputs/data/`：`model_metrics.json`、`segments_data.json`、`feature_importance.json` 等
- `algorithm/outputs/visualizations/`：分群图、客群画像图、分类评估图

## 6. GitHub 发布建议

- 本仓库包含较多历史文档、实验输出和演示模块，建议通过 `.gitignore` 管理构建产物与临时文件
- 如果你希望对外展示更聚焦，可优先在仓库首页强调 `algorithm/` 与 `api/`
- `backend(1)` 命名保留了当前本地结构，若对外发布可在后续重命名为 `backend`

## 7. 文档导航（评审入口）

### 根目录保留的核心说明

- `README.md`：仓库总览与启动方式
- `技术文档-更新版.md`：核心技术说明文档

### 模块导航索引

- `algorithm/INDEX.md`
- `api/INDEX.md`
- `backend(1)/INDEX.md`

### 过程文档归档

- `docs/archive/README.md`
- `docs/archive/` 下包含全部过程性中文报告与阶段复盘文档

## 8. 许可证与说明

- 仅用于学习、研究和竞赛交流
- 若包含业务敏感数据，请在发布前确认已脱敏
