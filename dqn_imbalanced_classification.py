"""
基于深度Q学习网络（DQN）的不平衡分类模型
参考论文: Deep Reinforcement Learning for Imbalanced Classification (2019)

核心思想:
1. 将分类问题形式化为马尔可夫决策过程（MDP）
2. 状态: 每个训练样本的特征向量
3. 动作: 分类决策（6个客群类别）
4. 奖励: 正确分类获得正奖励，错误分类获得负奖励（少数类奖励更高）
5. 通过DQN学习最优分类策略
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import deque
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# ============== 1. 马尔可夫决策过程（MDP）环境 ==============

class ClassificationEnvironment:
    """
    分类任务的仿真环境
    
    根据论文，将分类任务建模为序列决策过程：
    - 状态（State）: 用户特征向量
    - 动作（Action）: 分类决策（0-5，对应6个客群）
    - 奖励（Reward）: 基于分类正确性和类别频率的奖励函数
    """
    
    def __init__(self, X, y, class_weights=None):
        """
        初始化环境
        
        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            class_weights: 类别权重字典，用于计算奖励
        """
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        self.n_actions = len(np.unique(y))  # 动作空间大小 = 类别数
        
        # 计算类别权重（用于不平衡奖励设计）
        if class_weights is None:
            class_counts = np.bincount(y)
            # 少数类权重更高
            self.class_weights = self.n_samples / (self.n_actions * class_counts)
        else:
            self.class_weights = class_weights
        
        # 归一化权重
        self.class_weights = self.class_weights / self.class_weights.sum() * self.n_actions
        
        # 当前episode的状态
        self.current_idx = 0
        self.sample_indices = np.arange(self.n_samples)
        np.random.shuffle(self.sample_indices)
        
    def reset(self):
        """重置环境，开始新的episode"""
        self.current_idx = 0
        np.random.shuffle(self.sample_indices)
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态（特征向量）"""
        if self.current_idx >= self.n_samples:
            return None
        idx = self.sample_indices[self.current_idx]
        return self.X[idx], self.y[idx]
    
    def step(self, action):
        """
        执行动作，返回奖励和下一个状态
        
        参数:
            action: 分类决策 (0 到 n_actions-1)
            
        返回:
            next_state: 下一个状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        if self.current_idx >= self.n_samples:
            return None, 0, True, {}
        
        # 获取当前样本的真实标签
        idx = self.sample_indices[self.current_idx]
        true_label = self.y[idx]
        
        # 计算奖励（论文核心：少数类奖励更高）
        if action == true_label:
            # 正确分类：奖励 = 类别权重
            reward = self.class_weights[true_label]
        else:
            # 错误分类：惩罚 = -类别权重
            reward = -self.class_weights[true_label]
        
        # 移动到下一个样本
        self.current_idx += 1
        
        # 检查是否结束
        done = self.current_idx >= self.n_samples
        
        # 获取下一个状态
        if not done:
            next_state = self._get_state()
        else:
            next_state = None
        
        info = {
            'true_label': true_label,
            'predicted_label': action,
            'correct': action == true_label
        }
        
        return next_state, reward, done, info


# ============== 2. DQN网络架构 ==============

class DQN(nn.Module):
    """
    深度Q网络
    
    输入: 状态（特征向量）
    输出: Q值（每个动作的价值估计）
    """
    
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64]):
        """
        初始化DQN网络
        
        参数:
            input_dim: 输入特征维度
            output_dim: 输出动作维度（类别数）
            hidden_dims: 隐藏层维度列表
        """
        super(DQN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """前向传播"""
        return self.network(x)


# ============== 3. 经验回放缓冲区 ==============

class ReplayBuffer:
    """
    经验回放缓冲区
    
    存储 (state, action, reward, next_state, done) 转换
    用于打破样本相关性，提高训练稳定性
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样批次"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 处理next_states，可能包含None
        next_states_list = []
        for ns in next_states:
            if ns is not None:
                next_states_list.append(ns)
            else:
                next_states_list.append(np.zeros_like(states[0]))  # 用零填充
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states_list)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============== 4. DQN Agent ==============

class DQNAgent:
    """
    DQN智能体
    
    使用DQN算法学习分类策略
    """
    
    def __init__(self, state_dim, action_dim, device='cpu'):
        """
        初始化DQN Agent
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            device: 计算设备
        """
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 10  # 目标网络更新频率
        
        # 创建Q网络和目标网络
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # 训练统计
        self.episode_count = 0
        self.loss_history = []
        self.reward_history = []
        
    def select_action(self, state, training=True):
        """
        选择动作（ε-贪心策略）
        
        参数:
            state: 当前状态
            training: 是否处于训练模式
        """
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randrange(self.action_dim)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(1).item()
    
    def train_step(self):
        """
        训练步骤：从经验回放中采样并更新Q网络
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_states = next_states.to(self.device)
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============== 5. 训练函数 ==============

def train_dqn(agent, env, n_episodes=100, save_dir='outputs'):
    """
    训练DQN Agent
    
    参数:
        agent: DQN Agent
        env: 分类环境
        n_episodes: 训练轮数
        save_dir: 保存目录
    """
    print("\n" + "=" * 80)
    print("开始DQN训练")
    print("=" * 80)
    
    save_dir = Path(save_dir)
    
    for episode in range(n_episodes):
        # 重置环境
        state_tuple = env.reset()
        if state_tuple is None:
            break
        
        episode_reward = 0
        episode_loss = []
        correct_count = 0
        total_count = 0
        
        # Episode循环
        while True:
            state, true_label = state_tuple
            
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state_tuple, reward, done, info = env.step(action)
            
            # 存储经验
            if next_state_tuple is not None:
                next_state, _ = next_state_tuple
            else:
                next_state = None
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 训练
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            # 统计
            episode_reward += reward
            if info['correct']:
                correct_count += 1
            total_count += 1
            
            # 下一个状态
            if done:
                break
            state_tuple = next_state_tuple
        
        # 更新目标网络
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 记录统计
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        accuracy = correct_count / total_count if total_count > 0 else 0
        agent.loss_history.append(avg_loss)
        agent.reward_history.append(episode_reward)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Accuracy: {accuracy:.3f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("\n训练完成！")
    return agent


# ============== 6. 评估函数 ==============

def evaluate_dqn(agent, X_test, y_test, label_names, save_dir='outputs'):
    """
    评估DQN模型
    
    参数:
        agent: 训练好的DQN Agent
        X_test: 测试特征
        y_test: 测试标签
        label_names: 标签名称
        save_dir: 保存目录
    """
    print("\n" + "=" * 80)
    print("评估DQN模型")
    print("=" * 80)
    
    save_dir = Path(save_dir)
    
    # 预测
    y_pred = []
    for i in range(len(X_test)):
        state = X_test[i]
        action = agent.select_action(state, training=False)
        y_pred.append(action)
    
    y_pred = np.array(y_pred)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(label_names))
    )
    
    print(f"\n总体准确率: {accuracy:.4f}")
    print("\n各类别性能:")
    print(f"{'客群':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for i, name in enumerate(label_names):
        print(f"{name:<15} {precision[i]:<12.3f} {recall[i]:<12.3f} {f1[i]:<12.3f} {support[i]:<10}")
    
    # 保存详细报告
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    with open(save_dir / 'logs' / 'dqn_classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 保存性能指标
    metrics = {
        'accuracy': float(accuracy),
        'per_class_metrics': {
            label_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(label_names))
        },
        'macro_avg': {
            'precision': float(precision.mean()),
            'recall': float(recall.mean()),
            'f1_score': float(f1.mean())
        },
        'weighted_avg': {
            'precision': float(np.average(precision, weights=support)),
            'recall': float(np.average(recall, weights=support)),
            'f1_score': float(np.average(f1, weights=support))
        }
    }
    
    with open(save_dir / 'logs' / 'dqn_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    return y_pred, metrics


# ============== 7. 可视化函数 ==============

def plot_training_curves(agent, save_dir='outputs'):
    """绘制训练曲线"""
    save_dir = Path(save_dir)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # 损失曲线
    ax1.plot(agent.loss_history, linewidth=2, color='#E74C3C')
    ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('DQN训练损失曲线', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 奖励曲线
    ax2.plot(agent.reward_history, linewidth=2, color='#3498DB')
    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax2.set_title('DQN累积奖励曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'visualizations' / 'dqn_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] dqn_training_curves.png")


def plot_confusion_matrix(y_true, y_pred, label_names, save_dir='outputs'):
    """绘制混淆矩阵"""
    save_dir = Path(save_dir)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': '样本数'})
    plt.xlabel('预测标签', fontsize=12, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12, fontweight='bold')
    plt.title('DQN分类混淆矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'visualizations' / 'dqn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] dqn_confusion_matrix.png")


def plot_class_performance(metrics, label_names, save_dir='outputs'):
    """绘制各类别性能对比"""
    save_dir = Path(save_dir)
    
    per_class = metrics['per_class_metrics']
    
    precision = [per_class[name]['precision'] for name in label_names]
    recall = [per_class[name]['recall'] for name in label_names]
    f1 = [per_class[name]['f1_score'] for name in label_names]
    
    x = np.arange(len(label_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(x - width, precision, width, label='Precision', color='#3498DB')
    ax.bar(x, recall, width, label='Recall', color='#E74C3C')
    ax.bar(x + width, f1, width, label='F1-Score', color='#2ECC71')
    
    ax.set_xlabel('客群', fontsize=12, fontweight='bold')
    ax.set_ylabel('分数', fontsize=12, fontweight='bold')
    ax.set_title('DQN各客群分类性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'visualizations' / 'dqn_class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] dqn_class_performance.png")


# ============== 8. 主函数 ==============

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("基于深度Q学习网络（DQN）的不平衡分类模型")
    print("参考论文: Deep Reinforcement Learning for Imbalanced Classification (2019)")
    print("=" * 80)
    
    # 路径设置
    data_path = Path('algorithm/outputs/data/features_with_segments.csv')
    save_dir = Path('DQN/outputs')
    
    # 创建保存目录
    (save_dir / 'models').mkdir(parents=True, exist_ok=True)
    (save_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
    (save_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (save_dir / 'data').mkdir(parents=True, exist_ok=True)
    
    # ============== 数据加载 ==============
    print("\n[Step 1] 加载数据...")
    df = pd.read_csv(data_path)
    print(f"  数据形状: {df.shape}")
    print(f"  客群分布:\n{df['segment_name'].value_counts()}")
    
    # 提取特征和标签
    # 排除非特征列
    exclude_cols = ['USER_ID', 'segment_name', 'max_score', 'max_score_segment']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y_names = df['segment_name'].values
    
    # 标签编码
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_names)
    label_names = label_encoder.classes_.tolist()
    
    print(f"  特征维度: {X.shape[1]}")
    print(f"  类别数量: {len(label_names)}")
    print(f"  类别名称: {label_names}")
    
    # ============== 数据划分 ==============
    print("\n[Step 2] 划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    
    # ============== 特征标准化 ==============
    print("\n[Step 3] 特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ============== 创建环境 ==============
    print("\n[Step 4] 创建分类环境...")
    train_env = ClassificationEnvironment(X_train_scaled, y_train)
    print(f"  状态维度: {train_env.n_features}")
    print(f"  动作空间: {train_env.n_actions}")
    print(f"  类别权重: {dict(enumerate(train_env.class_weights))}")
    
    # ============== 创建DQN Agent ==============
    print("\n[Step 5] 创建DQN Agent...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    agent = DQNAgent(
        state_dim=train_env.n_features,
        action_dim=train_env.n_actions,
        device=device
    )
    print(f"  Q网络架构: {agent.policy_net}")
    
    # ============== 训练 ==============
    print("\n[Step 6] 训练DQN模型...")
    n_episodes = 100
    agent = train_dqn(agent, train_env, n_episodes=n_episodes, save_dir=save_dir)
    
    # 保存模型
    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'loss_history': agent.loss_history,
        'reward_history': agent.reward_history
    }, save_dir / 'models' / 'dqn_model.pth')
    print(f"  [OK] 模型已保存: dqn_model.pth")
    
    # ============== 评估 ==============
    print("\n[Step 7] 评估DQN模型...")
    y_pred, metrics = evaluate_dqn(agent, X_test_scaled, y_test, label_names, save_dir)
    
    # ============== 可视化 ==============
    print("\n[Step 8] 生成可视化...")
    plot_training_curves(agent, save_dir)
    plot_confusion_matrix(y_test, y_pred, label_names, save_dir)
    plot_class_performance(metrics, label_names, save_dir)
    
    # ============== 保存结果 ==============
    print("\n[Step 9] 保存结果...")
    results_df = pd.DataFrame({
        'true_label': [label_names[i] for i in y_test],
        'predicted_label': [label_names[i] for i in y_pred],
        'correct': y_test == y_pred
    })
    results_df.to_csv(save_dir / 'data' / 'dqn_predictions.csv', index=False, encoding='utf-8-sig')
    print(f"  [OK] 预测结果已保存: dqn_predictions.csv")
    
    # ============== 完成 ==============
    print("\n" + "=" * 80)
    print("DQN不平衡分类完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  [MODEL] {save_dir / 'models' / 'dqn_model.pth'}")
    print(f"  [DATA] {save_dir / 'data' / 'dqn_predictions.csv'}")
    print(f"  [LOG] {save_dir / 'logs' / 'dqn_metrics.json'}")
    print(f"  [LOG] {save_dir / 'logs' / 'dqn_classification_report.json'}")
    print(f"  [VIZ] {save_dir / 'visualizations' / 'dqn_training_curves.png'}")
    print(f"  [VIZ] {save_dir / 'visualizations' / 'dqn_confusion_matrix.png'}")
    print(f"  [VIZ] {save_dir / 'visualizations' / 'dqn_class_performance.png'}")
    
    print(f"\n总体准确率: {metrics['accuracy']:.4f}")
    print(f"宏平均 F1-Score: {metrics['macro_avg']['f1_score']:.4f}")
    print(f"加权平均 F1-Score: {metrics['weighted_avg']['f1_score']:.4f}")
    

if __name__ == '__main__':
    main()

