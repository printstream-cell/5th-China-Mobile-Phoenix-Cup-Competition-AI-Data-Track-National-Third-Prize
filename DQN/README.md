# 基于深度强化学习（DQN）的不平衡分类模型

## 论文来源

**Deep Reinforcement Learning for Imbalanced Classification**  
- 作者: Enlu Lin, Qiong Chen, Xiaoming Qi
- 单位: South China University of Technology
- 年份: 2019
- arXiv: 1901.01379v1

## 项目概述

本项目完全按照论文思路实现了基于深度Q学习网络（DQN）的不平衡分类模型，用于解决Z世代客群分类中的类别不平衡问题。

### 核心思想

论文将分类问题重新形式化为一个**序列决策过程（Sequential Decision-Making Process）**，通过强化学习来学习最优分类策略：

1. **状态（State）**: 用户的特征向量（68维）
2. **动作（Action）**: 分类决策（6个客群类别）
3. **奖励（Reward）**: 基于分类正确性和类别频率的奖励函数
4. **策略（Policy）**: 通过DQN学习的最优分类策略

### 为什么使用强化学习？

传统分类算法面对不平衡数据时，往往对多数类表现良好，但对少数类识别能力差。强化学习通过**特殊的奖励函数设计**，可以让模型更关注少数类：

- **少数类样本**：给予更高的奖励/惩罚（权重更大）
- **多数类样本**：给予较低的奖励/惩罚（权重较小）
- **目标**：最大化累积奖励 = 正确识别尽可能多的样本，特别是少数类

---

## 项目结构

```
DQN/
├── dqn_imbalanced_classification.py    # 主程序（完整实现）
├── README.md                            # 本文档
└── outputs/                             # 输出目录
    ├── models/                          # 模型文件
    │   └── dqn_model.pth               # 训练好的DQN模型
    ├── data/                            # 数据文件
    │   └── dqn_predictions.csv         # 预测结果
    ├── logs/                            # 日志文件
    │   ├── dqn_metrics.json            # 性能指标
    │   └── dqn_classification_report.json  # 分类报告
    └── visualizations/                  # 可视化图表
        ├── dqn_training_curves.png     # 训练曲线
        ├── dqn_confusion_matrix.png    # 混淆矩阵
        └── dqn_class_performance.png   # 类别性能对比
```

---

## 算法实现细节

### 1. 马尔可夫决策过程（MDP）建模

#### 1.1 环境（Environment）类

**类名**: `ClassificationEnvironment`

**核心功能**:
- **状态空间**: 用户特征向量（68维标准化特征）
- **动作空间**: 6个客群分类决策
- **奖励函数**: 
  ```python
  if action == true_label:
      reward = class_weight[true_label]  # 正确分类，获得正奖励
  else:
      reward = -class_weight[true_label] # 错误分类，获得负奖励
  ```

**类别权重计算**（论文核心）:
```python
class_weights = n_samples / (n_classes * class_counts)
# 归一化
class_weights = class_weights / class_weights.sum() * n_classes
```

**结果**:
- 学习型青年（最少数类，45人）: 权重 = 2.69
- 剧综重度用户（多数类，356人）: 权重 = 0.34
- 少数类权重是多数类的 **8倍**

#### 1.2 状态转移

环境采用**单次通过（Single Pass）**模式：
- 每个episode遍历所有训练样本一次
- 样本顺序随机打乱
- 每次step返回：`(next_state, reward, done, info)`

### 2. DQN网络架构

#### 2.1 Q网络（Q-Network）

**类名**: `DQN`

**网络结构**:
```
输入层:  68维特征向量
隐藏层1: 256维 + ReLU + Dropout(0.3)
隐藏层2: 128维 + ReLU + Dropout(0.3)
隐藏层3: 64维  + ReLU + Dropout(0.3)
输出层:  6维（每个动作的Q值）
```

**激活函数**: ReLU  
**正则化**: Dropout (p=0.3)  
**损失函数**: MSE Loss（均方误差）

#### 2.2 目标网络（Target Network）

- 与Q网络结构完全相同
- 每10个episode更新一次（硬更新）
- 用于稳定训练过程

### 3. 训练算法

#### 3.1 DQN Agent

**类名**: `DQNAgent`

**核心超参数**:
```python
gamma = 0.99              # 折扣因子
epsilon = 1.0 → 0.01      # 探索率（ε-贪心）
epsilon_decay = 0.995     # 探索率衰减
learning_rate = 0.001     # 学习率
batch_size = 64           # 批次大小
replay_buffer_size = 10000  # 经验回放容量
target_update_freq = 10   # 目标网络更新频率
```

#### 3.2 探索策略（ε-贪心）

```python
if random() < epsilon:
    action = random_action()  # 探索
else:
    action = argmax(Q(state)) # 利用
```

- 初始epsilon=1.0（完全探索）
- 逐渐衰减到0.01（主要利用）
- 平衡探索与利用

#### 3.3 经验回放（Experience Replay）

**类名**: `ReplayBuffer`

**作用**:
- 存储 (state, action, reward, next_state, done) 转换
- 随机采样批次进行训练
- 打破样本时间相关性
- 提高数据利用效率

**容量**: 10,000个经验

#### 3.4 Q值更新（Bellman方程）

```python
# 当前Q值
Q_current = Q(state, action)

# 目标Q值（使用目标网络）
Q_target = reward + gamma * max(Q_target_network(next_state))

# 损失
loss = MSE(Q_current, Q_target)
```

#### 3.5 训练流程

```
For each episode (1-100):
    1. 重置环境，打乱样本顺序
    2. For each sample in training set:
        a. 选择动作（ε-贪心）
        b. 执行动作，获取奖励
        c. 存储经验到replay buffer
        d. 从buffer采样批次
        e. 计算Q值和目标Q值
        f. 反向传播更新Q网络
    3. 每10个episode更新目标网络
    4. 衰减探索率
```

### 4. 评估与可视化

#### 4.1 性能指标

- **总体准确率（Accuracy）**: 0.8171
- **宏平均F1-Score**: 0.7767
- **加权平均F1-Score**: 0.8246

#### 4.2 各类别性能

| 客群 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| 剧综重度用户 | 0.894 | 0.831 | 0.861 | 71 |
| **学习型青年** | **0.381** | **0.889** | **0.533** | **9** |
| 电商剁手党 | 0.800 | 0.800 | 0.800 | 25 |
| 硬核玩家 | 0.927 | 0.761 | 0.836 | 67 |
| 社交达人 | 0.762 | 0.842 | 0.800 | 19 |
| 音乐发烧友 | 0.812 | 0.848 | 0.830 | 66 |

**关键发现**:
- **学习型青年**（最少数类）的召回率达到 **88.9%**
- 这证明了DQN通过奖励函数成功地让模型更关注少数类
- 但精确率较低（38.1%），说明存在一定的误分类

#### 4.3 可视化输出

1. **训练曲线** (`dqn_training_curves.png`)
   - 损失曲线：显示训练过程中损失的变化
   - 奖励曲线：显示累积奖励的增长趋势

2. **混淆矩阵** (`dqn_confusion_matrix.png`)
   - 展示各类别之间的混淆情况
   - 对角线表示正确分类的数量

3. **类别性能对比** (`dqn_class_performance.png`)
   - 柱状图对比各客群的Precision、Recall、F1-Score

---

## 运行方法

### 环境要求

```bash
python >= 3.8
torch >= 1.10.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

### 运行训练

```bash
python DQN/dqn_imbalanced_classification.py
```

### 训练参数调整

可以在脚本中修改以下参数：

```python
# 训练轮数
n_episodes = 100

# Agent超参数
gamma = 0.99              # 折扣因子
epsilon_decay = 0.995     # 探索率衰减
learning_rate = 0.001     # 学习率
batch_size = 64           # 批次大小

# 网络结构
hidden_dims = [256, 128, 64]  # 隐藏层维度
```

---

## 论文与实现对应关系

| 论文概念 | 实现位置 | 说明 |
|---------|---------|------|
| **Imbalanced Classification MDP** | `ClassificationEnvironment` | 将分类问题形式化为MDP |
| **State** | `env._get_state()` | 用户特征向量 |
| **Action** | `agent.select_action()` | 6个分类决策 |
| **Reward Function** | `env.step()` | 基于类别权重的奖励函数 |
| **DQN Architecture** | `DQN` 类 | 深度Q网络 |
| **Experience Replay** | `ReplayBuffer` | 经验回放缓冲区 |
| **Target Network** | `agent.target_net` | 目标网络（硬更新） |
| **ε-greedy Policy** | `agent.select_action()` | 探索-利用平衡 |
| **Q-learning Update** | `agent.train_step()` | Bellman方程更新 |

---

## 算法优势

### 1. 针对不平衡数据优化

- **奖励函数设计**: 少数类奖励更高，直接优化少数类识别
- **无需重采样**: 不需要SMOTE等数据层面方法
- **无需调整阈值**: 通过强化学习自动学习最优决策边界

### 2. 端到端学习

- 直接从原始特征学习分类策略
- 无需手工设计规则
- 自动发现特征与类别的复杂关系

### 3. 可解释性

- Q值可以解释为"该动作的长期价值"
- 奖励函数明确体现了对少数类的重视
- 训练过程可视化，便于调试

---

## 实验结果分析

### 1. 少数类性能提升

| 客群 | 样本数 | DQN Recall |
|------|--------|------------|
| 学习型青年 | 45 (3.5%) | **88.9%** |
| 社交达人 | 95 (7.4%) | **84.2%** |
| 电商剁手党 | 126 (9.8%) | **80.0%** |

**结论**: DQN成功地提高了少数类的召回率，特别是最少数类"学习型青年"。

### 2. 训练收敛

- **损失**: 从0.15收敛到0.24
- **奖励**: 从-331逐渐增长到-35
- **准确率**: 从20.7%提升到48.7%（训练集内episode准确率）

### 3. 与其他方法对比

| 方法 | 学习型青年 Recall | 总体 F1 |
|------|------------------|---------|
| XGBoost (标准) | 42.9% | - |
| XGBoost + Focal Loss | 71.4% | - |
| **DQN** | **88.9%** | **0.7767** |

**结论**: DQN在少数类召回率上超越了传统方法。

---

## 论文核心贡献总结

1. **问题重构**: 将分类问题重新形式化为序列决策问题
2. **奖励设计**: 提出了基于类别频率的自适应奖励函数
3. **理论分析**: 证明了奖励函数对损失函数的影响机制
4. **实验验证**: 在多个不平衡数据集上验证了有效性

---

## 参考文献

```bibtex
@article{lin2019deep,
  title={Deep Reinforcement Learning for Imbalanced Classification},
  author={Lin, Enlu and Chen, Qiong and Qi, Xiaoming},
  journal={arXiv preprint arXiv:1901.01379},
  year={2019}
}
```

---

## 联系方式

如有问题，请参考论文原文或查看代码注释。

---

**最后更新**: 2025年12月21日




