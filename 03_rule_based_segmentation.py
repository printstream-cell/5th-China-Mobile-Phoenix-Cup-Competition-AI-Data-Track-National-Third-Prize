"""
Z世代客群识别 - 基于规则的分群（软阈值 + 综合得分）

核心思想：
1. 从features_data.csv读取标准化后的特征（02_feature_engineering.py的输出）
2. 计算相对偏好特征（ratio）：APP使用天数占比
3. 综合打分：相对偏好（主条件）+ 绝对天数/行为特征（辅助条件）
4. 所有用户根据最高得分归入6大客群之一
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============== 配置路径 ==============
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "algorithm" / "outputs" / "data"
VIZ_DIR = BASE_DIR / "algorithm" / "outputs" / "visualizations" / "segmentation"
SEG_VIZ_DIR = BASE_DIR / "algorithm" / "outputs" / "visualizations" / "segments"

VIZ_DIR.mkdir(parents=True, exist_ok=True)
SEG_VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Z世代客群识别 - 软阈值 + 综合得分")
print("=" * 80)

# ============== Step 1: 加载数据 ==============
print("\n[Step 1] 加载特征数据...")

# 优先使用标准化后的特征（features_data.csv）
features_file = DATA_DIR / "features_data.csv"
if features_file.exists():
    print(f"  使用标准化特征: {features_file}")
    features_df = pd.read_csv(features_file)
else:
    print(f"  标准化特征不存在，使用原始数据: cleaned_data.csv")
    cleaned_df = pd.read_csv(DATA_DIR / "cleaned_data.csv")
    
    # 提取需要的原始特征
    features_df = pd.DataFrame()
    
    # APP使用天数（原始值，未标准化）
    app_fields = {
        'video_app_days': 'N3M_AVG_VIDEO_APP_USE_DAYS',
        'game_app_days': 'N3M_AVG_GAME_APP_USE_DAYS',
        'social_app_days': 'N3M_AVG_SOCIAL_APP_USE_DAYS',
        'learn_app_days': 'N3M_AVG_LEARN_APP_USE_DAYS',
        'music_app_days': 'N3M_AVG_MUSIC_APP_USE_DAYS',
        'shop_app_days': 'N3M_AVG_SHOP_APP_USE_DAYS',
        'read_app_days': 'N3M_AVG_READ_APP_USE_DAYS',
    }
    
    for feature_name, raw_field in app_fields.items():
        if raw_field in cleaned_df.columns:
            features_df[feature_name] = pd.to_numeric(cleaned_df[raw_field], errors='coerce').fillna(0)
        else:
            features_df[feature_name] = 0
    
    # 其他需要的原始特征
    features_df['arpu_current'] = pd.to_numeric(cleaned_df['DIS_ARPU'], errors='coerce').fillna(0)
    features_df['video_app_duration'] = pd.to_numeric(cleaned_df.get('N3M_AVG_VIDEO_APP_USE_DURA', 0), errors='coerce').fillna(0)
    features_df['video_intensity'] = features_df['video_app_duration'] / (features_df['video_app_days'] + 1)
    features_df['music_app_duration'] = pd.to_numeric(cleaned_df.get('N3M_AVG_MUSIC_APP_USE_DURA', 0), errors='coerce').fillna(0)
    features_df['music_intensity'] = features_df['music_app_duration'] / (features_df['music_app_days'] + 1)
    features_df['school_resident_days'] = pd.to_numeric(cleaned_df.get('T_school_resident', 0), errors='coerce').fillna(0)
    features_df['email_app_days'] = pd.to_numeric(cleaned_df.get('N3M_AVG_EMAIL_APP_USE_DAYS', 0), errors='coerce').fillna(0)
    features_df['finance_app_days'] = pd.to_numeric(cleaned_df.get('N3M_AVG_FINANC_APP_USE_DAYS', 0), errors='coerce').fillna(0)
    
    # 流量特征
    if 'day_flux' in cleaned_df.columns and 'night_flux' in cleaned_df.columns:
        day_flux = pd.to_numeric(cleaned_df['day_flux'], errors='coerce').fillna(0)
        night_flux = pd.to_numeric(cleaned_df['night_flux'], errors='coerce').fillna(0)
        total_flux = day_flux + night_flux + 1
        features_df['night_owl_score'] = night_flux / total_flux
    else:
        features_df['night_owl_score'] = 0.5
    
    # 保留USER_ID（如果有）
    if 'USER_ID' in cleaned_df.columns:
        features_df['USER_ID'] = cleaned_df['USER_ID']

print(f"  样本数: {len(features_df):,}")
print(f"  特征数: {len(features_df.columns)}")

# ============== Step 2: 计算相对偏好特征 ==============
print("\n[Step 2] 计算相对偏好特征...")

# 确保使用原始的APP天数（不是标准化的），重新计算ratio
# 注意：根据公式描述，C = {视频, 游戏, 社交, 学习, 音乐, 购物} - 6个核心类别
# read_app_days不参与total_app_days的计算，但可以单独使用
core_app_cols = ['video_app_days', 'game_app_days', 'social_app_days', 
                 'learn_app_days', 'music_app_days', 'shop_app_days']

# 所有APP列（包括read，用于其他用途）
all_app_cols = core_app_cols + ['read_app_days']

# 如果加载的是标准化特征，需要反标准化APP天数列
# 或者直接从cleaned_data.csv重新读取原始APP天数
if features_file.exists():
    # 从cleaned_data.csv重新读取原始APP天数
    print("  [WARNING] features_data.csv包含标准化特征，正在从cleaned_data.csv重新加载原始APP天数...")
    cleaned_df = pd.read_csv(DATA_DIR / "cleaned_data.csv")
    
    # 重新提取原始APP天数，覆盖标准化的值
    app_fields = {
        'video_app_days': 'N3M_AVG_VIDEO_APP_USE_DAYS',
        'game_app_days': 'N3M_AVG_GAME_APP_USE_DAYS',
        'social_app_days': 'N3M_AVG_SOCIAL_APP_USE_DAYS',
        'learn_app_days': 'N3M_AVG_LEARN_APP_USE_DAYS',
        'music_app_days': 'N3M_AVG_MUSIC_APP_USE_DAYS',
        'shop_app_days': 'N3M_AVG_SHOP_APP_USE_DAYS',
        'read_app_days': 'N3M_AVG_READ_APP_USE_DAYS',
    }
    
    for feature_name, raw_field in app_fields.items():
        if raw_field in cleaned_df.columns:
            features_df[feature_name] = pd.to_numeric(cleaned_df[raw_field], errors='coerce').fillna(0)
    
    # 同样重新提取ARPU（原始值）
    if 'DIS_ARPU' in cleaned_df.columns:
        features_df['arpu_current'] = pd.to_numeric(cleaned_df['DIS_ARPU'], errors='coerce').fillna(0)
    
    print("  [OK] 已重新加载原始APP天数和ARPU")

# 计算ratio特征（APP使用天数占比）
# 根据公式：Rc = Dc / (Dtotal + ε)
# 其中 Dtotal = sum(Dc) for c in C
# C = {视频, 游戏, 社交, 学习, 音乐, 购物} - 6个核心类别
# read_app_days不参与total_app_days的计算，但可以单独计算read_ratio

# 计算核心6类APP的总天数（不包含read_app_days）
features_df['total_app_days'] = features_df[core_app_cols].sum(axis=1)

# 计算6个核心类别的相对偏好
for app in ['video', 'game', 'social', 'learn', 'music', 'shop']:
    days_col = f'{app}_app_days'
    if days_col in features_df.columns:
        # Rc = Dc / (Dtotal + ε)，其中Dtotal只包含6个核心类别
        features_df[f'{app}_ratio'] = features_df[days_col] / (features_df['total_app_days'] + 1e-6)
    else:
        features_df[f'{app}_ratio'] = 0

# 单独计算read_ratio（如果需要，可以基于包含read的total，或单独处理）
# 这里我们选择：read_ratio基于包含read的total_app_days_all
if 'read_app_days' in features_df.columns:
    total_app_days_all = features_df[all_app_cols].sum(axis=1)
    features_df['read_ratio'] = features_df['read_app_days'] / (total_app_days_all + 1e-6)
else:
    features_df['read_ratio'] = 0

print(f"  [OK] 已计算相对偏好特征（ratio），范围0-1")
print(f"  [INFO] total_app_days仅包含6个核心类别（视频、游戏、社交、学习、音乐、购物）")
print(f"  [INFO] read_app_days不参与核心相对偏好计算，但单独计算read_ratio")

# ============== Step 2.5: 相对偏好分段映射（分位数分箱） ==============
print("\n[Step 2.5] 相对偏好分段映射（分位数分箱）...")
print("  核心思想：将0附近的连续值转换为有业务含义的等级信号")

# 对每个APP类别的ratio进行分位数分箱
# 映射规则：
# - >= P75（前25%）: 1.0（高偏好）
# - >= P50（前50%）: 0.7（中等偏好）
# - >= P25（前75%）: 0.4（低偏好）
# - < P25（后25%）:  0.1（极低偏好）
#
# 特殊处理：对于极低频APP（P75=0），使用"是否使用"二分法

ratio_mapping_thresholds = {}

for app in ['video', 'game', 'social', 'learn', 'music', 'shop', 'read']:
    ratio_col = f'{app}_ratio'
    mapped_col = f'{app}_ratio_mapped'
    
    if ratio_col in features_df.columns:
        # 计算分位数阈值
        p25 = features_df[ratio_col].quantile(0.25)
        p50 = features_df[ratio_col].quantile(0.50)
        p75 = features_df[ratio_col].quantile(0.75)
        p90 = features_df[ratio_col].quantile(0.90)  # 额外计算P90用于极低频APP
        
        # 保存阈值（用于报告和解释）
        ratio_mapping_thresholds[app] = {
            'P25': float(p25),
            'P50': float(p50),
            'P75': float(p75),
            'P90': float(p90),
            'strategy': 'normal' if p75 > 0 else 'low_frequency'
        }
        
        # 判断是否为极低频APP：如果P75=0，说明超过75%的人不使用
        if p75 == 0:
            # 极低频APP：采用"是否使用"+"使用程度"混合策略
            # P90 > 0说明前10%的人在使用，进一步区分
            def map_ratio_low_freq(r):
                if r >= p90 and p90 > 0:
                    return 1.0  # 前10%，真正的高频使用者
                elif r > 0:
                    return 0.7  # 10%-100%，有使用但不多
                else:
                    return 0.1  # 完全不使用
            
            features_df[mapped_col] = features_df[ratio_col].apply(map_ratio_low_freq)
            print(f"  {app}: P90={p90:.4f} (极低频APP，使用二分法+P90)")
        else:
            # 正常频率APP：使用标准分位数分箱
            def map_ratio_normal(r):
                if r >= p75:
                    return 1.0  # 前25%，高偏好
                elif r >= p50:
                    return 0.7  # 前25%-50%，中等偏好
                elif r >= p25:
                    return 0.4  # 前50%-75%，低偏好
                else:
                    return 0.1  # 后25%，极低偏好
            
            features_df[mapped_col] = features_df[ratio_col].apply(map_ratio_normal)
            print(f"  {app}: P25={p25:.4f}, P50={p50:.4f}, P75={p75:.4f}")

print(f"  [OK] 已完成分位数分箱映射，生成*_ratio_mapped特征")
print(f"  [INFO] 正常频率APP：≥P75→1.0, ≥P50→0.7, ≥P25→0.4, <P25→0.1")
print(f"  [INFO] 极低频APP：≥P90→1.0, >0→0.7, =0→0.1")

# 保存映射阈值
mapping_thresholds_file = DATA_DIR / "ratio_mapping_thresholds.json"
with open(mapping_thresholds_file, 'w', encoding='utf-8') as f:
    json.dump(ratio_mapping_thresholds, f, ensure_ascii=False, indent=2)
print(f"  [OK] 映射阈值已保存到 ratio_mapping_thresholds.json")

# ============== Step 3: 计算数据驱动阈值 ==============
print("\n[Step 3] 计算数据驱动阈值...")

# 直接从原始特征计算阈值（中位数、75%分位数等）
thresholds = {
    'video_median': features_df['video_app_days'].median(),
    'video_75': features_df['video_app_days'].quantile(0.75),
    'game_median': features_df['game_app_days'].median(),
    'game_75': features_df['game_app_days'].quantile(0.75),
    'social_median': features_df['social_app_days'].median(),
    'social_75': features_df['social_app_days'].quantile(0.75),
    'learn_median': features_df['learn_app_days'].median(),
    'learn_75': features_df['learn_app_days'].quantile(0.75),
    'music_median': features_df['music_app_days'].median(),
    'music_75': features_df['music_app_days'].quantile(0.75),
    'shop_median': features_df['shop_app_days'].median(),
    'shop_75': features_df['shop_app_days'].quantile(0.75),
    'arpu_median': features_df['arpu_current'].median(),
    'arpu_75': features_df['arpu_current'].quantile(0.75),
    'night_owl_median': features_df['night_owl_score'].median(),
    'night_owl_75': features_df['night_owl_score'].quantile(0.75),
}

# 保存阈值
thresholds_file = DATA_DIR / "segmentation_thresholds.json"
with open(thresholds_file, 'w', encoding='utf-8') as f:
    json.dump(thresholds, f, ensure_ascii=False, indent=2)

print(f"  [OK] 已计算{len(thresholds)}个阈值并保存到 segmentation_thresholds.json")

# ============== Step 4: 定义得分函数 ==============
print("\n[Step 4] 定义得分函数...")

def score_drama_heavy(row, thresholds):
    """剧综重度用户得分（基于分箱映射后的相对偏好）"""
    score = 0.0
    
    # 主条件：视频分箱映射偏好（权重50%）
    score += row.get('video_ratio_mapped', 0) * 0.5
    
    # 辅助1：视频绝对使用天数（权重25%）
    if row['video_app_days'] > thresholds['video_75']:
        score += 0.25
    elif row['video_app_days'] > thresholds['video_median']:
        score += 0.15
    
    # 辅助2：视频使用强度（权重25%）
    if 'video_intensity' in row and row['video_intensity'] > 0:
        score += 0.25
    
    return score


def score_hardcore_gamer(row, thresholds):
    """硬核玩家得分（基于分箱映射后的相对偏好）"""
    score = 0.0
    
    # 主条件：游戏分箱映射偏好（权重50%）
    score += row.get('game_ratio_mapped', 0) * 0.5
    
    # 辅助1：游戏绝对天数（权重20%）
    if row['game_app_days'] > thresholds['game_median']:
        score += 0.20
    
    # 辅助2：深夜活跃度（权重20%）
    if row['night_owl_score'] > thresholds['night_owl_75']:
        score += 0.20
    
    # 辅助3：5G用户（权重10%）
    if 'is_5g' in row and row['is_5g'] == 1:
        score += 0.10
    
    return score


def score_social_master(row, thresholds):
    """社交达人得分（基于分箱映射后的相对偏好）"""
    score = 0.0
    
    # 主条件：社交分箱映射偏好（权重40%）
    score += row.get('social_ratio_mapped', 0) * 0.4
    
    # 辅助1：社交绝对天数（权重25%）
    if row['social_app_days'] > thresholds['social_75']:
        score += 0.25
    
    # 辅助2：通话频繁（权重20%）
    if 'CALL_DAYS' in row and row['CALL_DAYS'] > thresholds.get('call_days_75', 0):
        score += 0.20
    
    # 辅助3：社交圈广（权重15%）
    if 'DIFF_CALL_USER_CNT' in row and row['DIFF_CALL_USER_CNT'] > thresholds.get('diff_call_user_75', 0):
        score += 0.15
    
    return score


def score_studious_youth(row, thresholds):
    """学习型青年得分（基于分箱映射后的相对偏好）
    
    示例：展示分箱映射的威力
    - 原始ratio: 0.006 vs 0.012，模型难以区分
    - 映射后：0.1（<P25）vs 0.7（≥P50），等级差异明显
    """
    score = 0.0
    
    # 主条件：学习分箱映射偏好（权重40%）
    score += row.get('learn_ratio_mapped', 0) * 0.4
    
    # 核心2：阅读分箱映射偏好（权重30%）
    score += row.get('read_ratio_mapped', 0) * 0.3
    
    # 辅助1：邮件APP（权重15%）
    if 'email_app_days' in row and row['email_app_days'] > thresholds.get('email_median', 0):
        score += 0.15
    
    # 辅助2：办公APP（权重15%）
    if 'office_app_days' in row and row.get('office_app_days', 0) > 0:
        score += 0.15
    
    return score


def score_shopping_enthusiast(row, thresholds):
    """电商剁手党得分（基于分箱映射后的相对偏好+购物频次）
    购物看频次，不看时长
    """
    score = 0.0
    
    # 主条件：购物分箱映射偏好（权重30%，降低因为购物不看时长）
    score += row.get('shop_ratio_mapped', 0) * 0.3
    
    # 核心1：购物绝对天数（权重35%，购物看频次！）
    if row['shop_app_days'] > thresholds['shop_75']:
        score += 0.35
    elif row['shop_app_days'] > thresholds['shop_median']:
        score += 0.20
    elif row['shop_app_days'] > 0:
        score += 0.10
    
    # 核心2：消费能力（ARPU）（权重20%）
    if row['arpu_current'] > thresholds['arpu_75']:
        score += 0.20
    elif row['arpu_current'] > thresholds['arpu_median']:
        score += 0.10
    
    # 辅助：金融APP（权重15%）
    if 'finance_app_days' in row and row['finance_app_days'] > 0:
        score += 0.15
    
    return score


def score_music_lover(row, thresholds):
    """音乐发烧友得分（基于分箱映射后的相对偏好）"""
    score = 0.0
    
    # 主条件：音乐分箱映射偏好（权重50%）
    score += row.get('music_ratio_mapped', 0) * 0.5
    
    # 辅助1：音乐绝对天数（权重20%）
    if row['music_app_days'] > thresholds['music_median']:
        score += 0.20
    
    # 辅助2：音乐使用强度（权重15%）
    if 'music_intensity' in row and row['music_intensity'] > 0:
        score += 0.15
    
    # 辅助3：音乐使用时长（权重15%）
    if 'music_app_duration' in row and row['music_app_duration'] > 0:
        score += 0.15
    
    return score


# ============== Step 5: 计算得分 ==============
print("\n[Step 5] 计算微调后的得分...")

features_df['score_drama'] = features_df.apply(lambda row: score_drama_heavy(row, thresholds), axis=1)
features_df['score_gamer'] = features_df.apply(lambda row: score_hardcore_gamer(row, thresholds), axis=1)
features_df['score_social'] = features_df.apply(lambda row: score_social_master(row, thresholds), axis=1)
features_df['score_study'] = features_df.apply(lambda row: score_studious_youth(row, thresholds), axis=1)
features_df['score_shop'] = features_df.apply(lambda row: score_shopping_enthusiast(row, thresholds), axis=1)
features_df['score_music'] = features_df.apply(lambda row: score_music_lover(row, thresholds), axis=1)

score_cols = ['score_drama', 'score_gamer', 'score_social', 'score_study', 'score_shop', 'score_music']

features_df['max_score'] = features_df[score_cols].max(axis=1)
features_df['max_score_segment'] = features_df[score_cols].idxmax(axis=1)

segment_name_map = {
    'score_drama': '剧综重度用户',
    'score_gamer': '硬核玩家',
    'score_social': '社交达人',
    'score_study': '学习型青年',
    'score_shop': '电商剁手党',
    'score_music': '音乐发烧友'
}

features_df['segment_name'] = features_df['max_score_segment'].map(segment_name_map)

# ============== Step 6: 最终客群分配 ==============
print("\n[Step 6] 最终客群分配...")

# 所有用户根据最高得分归入对应客群（不使用任何强制保护）
print(f"  [OK] 所有用户根据最高得分归入对应客群")

segment_counts = features_df['segment_name'].value_counts()
print(f"\n  客群分布:")
for segment, count in segment_counts.items():
    pct = count / len(features_df) * 100
    print(f"    {segment}: {count}人 ({pct:.1f}%)")

# ============== Step 7: 分析客群特征 ==============
print("\n[Step 7] 分析各客群特征...")

segment_profiles = {}

for segment_name in segment_counts.index:
    segment_data = features_df[features_df['segment_name'] == segment_name]
    
    profile = {
        '样本数': int(len(segment_data)),
        '占比': f"{len(segment_data)/len(features_df)*100:.1f}%",
        'APP相对偏好': {},
        'APP绝对使用': {},
        '平均得分': {},
        '行为特征': {}
    }
    
    for app in ['video', 'game', 'social', 'learn', 'music', 'shop']:
        ratio_col = f'{app}_ratio'
        days_col = f'{app}_app_days'
        profile['APP相对偏好'][app] = {
            '均值': float(segment_data[ratio_col].mean()),
            '中位数': float(segment_data[ratio_col].median())
        }
        profile['APP绝对使用'][app] = {
            '均值': float(segment_data[days_col].mean()),
            '中位数': float(segment_data[days_col].median())
        }
    
    for score_col in score_cols:
        profile['平均得分'][score_col.replace('score_', '')] = float(segment_data[score_col].mean())
    
    profile['行为特征']['ARPU均值'] = float(segment_data['arpu_current'].mean())
    profile['行为特征']['夜猫子分数'] = float(segment_data['night_owl_score'].mean())
    profile['行为特征']['总APP使用天数'] = float(segment_data['total_app_days'].mean())
    
    if 'is_5g' in segment_data.columns:
        profile['行为特征']['5G用户占比'] = f"{segment_data['is_5g'].mean()*100:.1f}%"
    
    segment_profiles[segment_name] = profile

profile_file = DATA_DIR / "segment_profiles.json"
with open(profile_file, 'w', encoding='utf-8') as f:
    json.dump(segment_profiles, f, ensure_ascii=False, indent=2)
print(f"  [OK] segment_profiles.json已保存")

# ============== Step 8: 生成客群详细数据（segments_data.json） ==============
print("\n[Step 8] 生成客群详细数据 segments_data.json ...")

segments_info = []
segment_id_map = {
    1: '剧综重度用户',
    2: '硬核玩家',
    3: '社交达人',
    4: '学习型青年',
    5: '电商剁手党',
    6: '音乐发烧友'
}

# 加载原始人口学与ARPU信息（用于生成segments_data.json）
# 注意：使用features_df作为用户总数的基准（就是我们分群的1284个用户）
total_users = len(features_df)

try:
    cleaned_full_df = pd.read_csv(DATA_DIR / "cleaned_data.csv")
    has_demo = True
except FileNotFoundError:
    has_demo = False
    cleaned_full_df = None

# 如果需要人口学信息，进行left join（以features_df为主）
if has_demo and 'USER_ID' in features_df.columns and 'USER_ID' in cleaned_full_df.columns:
    demo_df = cleaned_full_df[['USER_ID', 'AGE', 'DIS_ARPU', 'CITY']].copy()
else:
    demo_df = None

print(f"  [INFO] 用户总数（分群基准）: {total_users}")

for seg_id in range(1, 7):
    seg_name = segment_name_map[f'score_{"drama" if seg_id == 1 else "gamer" if seg_id == 2 else "social" if seg_id == 3 else "study" if seg_id == 4 else "shop" if seg_id == 5 else "music"}']
    seg_rows = features_df[features_df['segment_name'] == seg_name]
    count = len(seg_rows)
    percentage = float(f"{(count / total_users * 100):.1f}") if total_users > 0 else 0.0

    # 如果有人口学信息，merge获取AGE、ARPU、CITY
    if has_demo and demo_df is not None and 'USER_ID' in seg_rows.columns:
        seg_rows_with_demo = pd.merge(seg_rows, demo_df, on='USER_ID', how='left')
        avg_age = float(seg_rows_with_demo['AGE'].mean()) if 'AGE' in seg_rows_with_demo.columns else None
        avg_arpu = float(seg_rows_with_demo['DIS_ARPU'].mean()) if 'DIS_ARPU' in seg_rows_with_demo.columns else None
        
        # 年龄分布
        age_distribution = {}
        if 'AGE' in seg_rows_with_demo.columns:
            age_counts = seg_rows_with_demo['AGE'].value_counts().sort_index()
            age_distribution = {int(k): int(v) for k, v in age_counts.items()}
        
        # 省份/城市分布
        province_distribution = {}
        if 'CITY' in seg_rows_with_demo.columns:
            city_counts = seg_rows_with_demo['CITY'].value_counts()
            province_distribution = {str(k): int(v) for k, v in city_counts.items()}
    else:
        avg_age = None
        avg_arpu = None
        age_distribution = {}
        province_distribution = {}

    # APP偏好与使用情况（来自segment_profiles）
    prof = segment_profiles.get(seg_name, {})
    app_prefs = {}
    for app_key, app_cn in [('video', '视频'), ('game', '游戏'), ('social', '社交'),
                            ('learn', '学习'), ('shop', '电商'), ('music', '音乐')]:
        app_pref = prof.get('APP相对偏好', {}).get(app_key, {})
        app_abs = prof.get('APP绝对使用', {}).get(app_key, {})
        app_prefs[app_key] = {
            "days_mean": app_abs.get('均', None),
            "days_median": app_abs.get('中位数', None),
            "ratio_mean": app_pref.get('均值', None),
            "ratio_median": app_pref.get('中位数', None),
            "label": app_cn
        }

    behavior = {
        "avg_arpu": prof.get('行为特征', {}).get('ARPU均值', None),
        "avg_night_owl_score": prof.get('行为特征', {}).get('夜猫子分数', None),
        "avg_total_app_days": prof.get('行为特征', {}).get('总APP使用天数', None)
    }

    segments_info.append({
        "segment_id": seg_id,
        "segment_name": seg_name,
        "user_count": int(count),
        "percentage": percentage,
        "avg_age": avg_age,
        "avg_arpu": avg_arpu,
        "demographics": {
            "age_distribution": age_distribution,
            "province_distribution": province_distribution
        },
        "app_preferences": app_prefs,
        "behavior_patterns": behavior
    })

segments_output = {
    "total_users": int(total_users),
    "segments": [
        {
            "id": seg["segment_id"],
            "name": seg["segment_name"],
            "count": seg["user_count"],
            "percentage": seg["percentage"],
            "avg_age": seg["avg_age"],
            "avg_arpu": seg["avg_arpu"],
            "demographics": seg["demographics"],
            "app_preferences": seg["app_preferences"],
            "behavior_patterns": seg["behavior_patterns"]
        }
        for seg in segments_info
    ]
}

segments_data_file = DATA_DIR / "segments_data.json"
with open(segments_data_file, 'w', encoding='utf-8') as f:
    json.dump(segments_output, f, ensure_ascii=False, indent=2)
print(f"  [OK] segments_data.json已保存: {segments_data_file}")

# ============== Step 9: 保存分类结果 ==============
print("\n[Step 9] 保存分类结果...")

# 9.1 保存features_with_segments.csv（用于XGBoost训练）
# 包含所有特征（除了USER_ID，如果有的话单独处理）
exclude_cols = ['USER_ID']  # USER_ID单独处理
feature_cols_for_output = [col for col in features_df.columns if col not in exclude_cols]

# 确保包含所有生成的特征
output_df = features_df[feature_cols_for_output].copy()

# 如果有USER_ID，加入USER_ID列（放在最前面）
if 'USER_ID' in features_df.columns:
    output_df = pd.concat([features_df[['USER_ID']], output_df], axis=1)

output_file = DATA_DIR / "features_with_segments.csv"
output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"  [OK] features_with_segments.csv已保存（用于XGBoost训练）")

# 9.2 输出带标签的cleaned_data.csv（在原始cleaned_data后追加segment_name列）
if has_demo and cleaned_full_df is not None and 'USER_ID' in features_df.columns:
    # 合并cleaned_data和segment_name
    cleaned_with_labels = pd.merge(
        cleaned_full_df,
        features_df[['USER_ID', 'segment_name']],
        on='USER_ID',
        how='left'  # 保留所有cleaned_data的用户（包括非Z世代）
    )
    # 对于非Z世代用户，segment_name会是NaN，可以填充为"非Z世代"
    cleaned_with_labels['segment_name'] = cleaned_with_labels['segment_name'].fillna('非Z世代')
    
    cleaned_labeled_file = DATA_DIR / "cleaned_data_with_labels.csv"
    cleaned_with_labels.to_csv(cleaned_labeled_file, index=False, encoding='utf-8-sig')
    print(f"  [OK] cleaned_data_with_labels.csv已保存（原始数据+客群标签）")
else:
    print(f"  [SKIP] 无法生成cleaned_data_with_labels.csv（缺少USER_ID或cleaned_data.csv）")

# 9.3 生成完整的feature_names.json（所有特征的说明）
# 统计实际特征数（不包括USER_ID）
actual_feature_cols = [col for col in features_df.columns if col != 'USER_ID']
feature_names_info = {
    "total_features": len(actual_feature_cols),
    "feature_categories": {
        "原始APP使用天数": ['video_app_days', 'game_app_days', 'social_app_days', 'learn_app_days', 'music_app_days', 'shop_app_days', 'read_app_days'],
        "相对偏好特征": ['video_ratio', 'game_ratio', 'social_ratio', 'learn_ratio', 'music_ratio', 'shop_ratio', 'read_ratio'],
        "得分特征": list(score_cols),
        "行为特征": ['arpu_current', 'night_owl_score', 'total_app_days', 'video_intensity', 'music_intensity', 'video_app_duration', 'music_app_duration'],
        "场景特征": ['school_resident_days', 'email_app_days', 'finance_app_days'],
        "分群结果": ['segment_name', 'max_score', 'max_score_segment']
    },
    "feature_descriptions": {
        "video_app_days": "视频APP平均使用天数/月",
        "game_app_days": "游戏APP平均使用天数/月",
        "social_app_days": "社交APP平均使用天数/月",
        "learn_app_days": "学习APP平均使用天数/月",
        "music_app_days": "音乐APP平均使用天数/月",
        "shop_app_days": "购物APP平均使用天数/月",
        "read_app_days": "阅读APP平均使用天数/月",
        "video_ratio": "视频APP使用天数占比（相对偏好）",
        "game_ratio": "游戏APP使用天数占比（相对偏好）",
        "social_ratio": "社交APP使用天数占比（相对偏好）",
        "learn_ratio": "学习APP使用天数占比（相对偏好）",
        "music_ratio": "音乐APP使用天数占比（相对偏好）",
        "shop_ratio": "购物APP使用天数占比（相对偏好）",
        "arpu_current": "当月ARPU（元）",
        "night_owl_score": "深夜活跃度（深夜流量占比）",
        "total_app_days": "所有APP使用总天数",
        "segment_name": "客群名称（规则分群结果）",
        "max_score": "最高得分",
        "score_drama": "剧综重度用户得分",
        "score_gamer": "硬核玩家得分",
        "score_social": "社交达人得分",
        "score_study": "学习型青年得分",
        "score_shop": "电商剁手党得分",
        "score_music": "音乐发烧友得分"
    }
}

feature_names_file = DATA_DIR / "feature_names.json"
with open(feature_names_file, 'w', encoding='utf-8') as f:
    json.dump(feature_names_info, f, ensure_ascii=False, indent=2)
print(f"  [OK] feature_names.json已保存（完整特征说明）")

# 9.4 保存segmentation_summary.json
summary = {
    "分类方法": "软阈值 + 综合得分",
    "样本总数": int(len(features_df)),
    "客群分布": {seg: int(count) for seg, count in segment_counts.items()},
    "客群占比": {seg: f"{count/len(features_df)*100:.1f}%" for seg, count in segment_counts.items()},
    "总用户数": int(len(features_df)),
    "核心逻辑": [
        "直接从cleaned_data.csv读取原始数据（不使用标准化特征）",
        "计算相对偏好：APP使用天数占比（ratio）",
        "综合打分：相对偏好（主条件60%）+ 绝对天数/行为特征（辅助条件40%）",
        "所有用户根据最高得分归入6大客群之一"
    ],
    "权重设置": {
        "剧综重度用户": "video_ratio权重60%，绝对天数>75%位15%",
        "硬核玩家": "game_ratio权重40%，深夜活跃度>75%位20%",
        "社交达人": "social_ratio权重35%，社交天数>75%位25%",
        "学习型青年": "learn+read_ratio权重40%，学校驻留30%",
        "电商剁手党": "shop_ratio权重60%，ARPU>75%位15%",
        "音乐发烧友": "music_ratio权重40%，绝对天数>中位数20%"
    }
}

summary_file = DATA_DIR / "segmentation_summary.json"
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"  [OK] segmentation_summary.json已保存")

# ============== Step 10: 生成可视化 ==============
print("\n[Step 10] 生成可视化...")

all_segments = segment_counts
colors = plt.cm.Set3(range(len(all_segments)))

# 图1: 客群分布饼图
print("  生成图1: segment_distribution_pie.png...")
fig, ax = plt.subplots(figsize=(10, 8))

wedges, texts, autotexts = ax.pie(
    all_segments.values,
    labels=all_segments.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 11}
)

for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

ax.set_title(f'Z世代6大客群分布（全量用户，n={len(features_df)}）',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(VIZ_DIR / "segment_distribution_pie.png", dpi=150, bbox_inches='tight')
plt.close()
print("    [OK] segment_distribution_pie.png")

# 图2: APP相对偏好热力图
print("  生成图2: app_preference_heatmap.png...")

all_segment_names = list(segment_counts.index)

heatmap_data = []
heatmap_labels = []

# 使用与雷达图相同的顺序，保持一致性
apps_order_for_heatmap = ['video', 'game', 'social', 'learn', 'shop', 'music']
apps_labels_for_heatmap = ['视频', '游戏', '社交', '学习', '电商', '音乐']

for segment_name in all_segment_names:
    profile = segment_profiles[segment_name]
    row = []
    for app in apps_order_for_heatmap:
        row.append(profile['APP相对偏好'][app]['均值'])
    heatmap_data.append(row)
    heatmap_labels.append(f"{segment_name}\n({profile['样本数']}人)")

heatmap_df = pd.DataFrame(heatmap_data, 
                          columns=apps_labels_for_heatmap,
                          index=heatmap_labels)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', 
            cbar_kws={'label': '相对偏好（占比）'}, ax=ax, linewidths=0.5)
ax.set_title('Z世代6大客群APP偏好热力图', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('APP类型', fontsize=12, fontweight='bold')
ax.set_ylabel('客群', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(VIZ_DIR / "app_preference_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("    [OK] app_preference_heatmap.png")

# ============== Step 10.5: 生成分位数映射可视化 ==============
print("\n[Step 10.5] 生成分位数映射可视化...")

# 读取映射阈值
with open(mapping_thresholds_file, 'r', encoding='utf-8') as f:
    mapping_thresholds = json.load(f)

# 创建综合可视化：每个APP的原始ratio分布 + 分位数阈值 + 映射后分布
fig = plt.figure(figsize=(20, 12))
fig.suptitle('相对偏好分位数映射可视化', fontsize=20, fontweight='bold', y=0.995)

apps_for_viz = ['video', 'game', 'social', 'learn', 'music', 'shop']
apps_labels_viz = ['视频', '游戏', '社交', '学习', '音乐', '电商']

for idx, (app, app_label) in enumerate(zip(apps_for_viz, apps_labels_viz)):
    ratio_col = f'{app}_ratio'
    mapped_col = f'{app}_ratio_mapped'
    
    if ratio_col not in features_df.columns:
        continue
    
    # 左子图：原始ratio分布 + 分位数阈值
    ax1 = plt.subplot(3, 4, idx * 2 + 1)
    
    ratio_data = features_df[ratio_col].values
    ratio_data = ratio_data[ratio_data >= 0]  # 只显示非负值
    
    # 使用渐变色板绘制直方图
    n, bins, patches = ax1.hist(ratio_data, bins=50, alpha=0.7, color='#0098CE', edgecolor='white', linewidth=0.3)
    
    # 标注分位数阈值
    thresholds_info = mapping_thresholds.get(app, {})
    strategy = thresholds_info.get('strategy', 'normal')
    
    # 渐变色板配色方案
    gradient_colors = {
        'p25': '#2F74B7',  # 深蓝
        'p50': '#0098CE',  # 青色
        'p75': '#00B9C8',  # 青绿色
        'p90': '#00D5AB',  # 薄荷绿
        'map_0.1': '#2F74B7',  # 深蓝（最低）
        'map_0.4': '#0098CE',  # 青色
        'map_0.7': '#00B9C8',  # 青绿色
        'map_1.0': '#00D5AB'   # 薄荷绿（最高）
    }
    
    if strategy == 'normal':
        p25 = thresholds_info.get('P25', 0)
        p50 = thresholds_info.get('P50', 0)
        p75 = thresholds_info.get('P75', 0)
        
        # 绘制分位数线（使用渐变色板）
        ax1.axvline(p25, color=gradient_colors['p25'], linestyle='--', linewidth=2.5, label=f'P25={p25:.4f}', alpha=0.8)
        ax1.axvline(p50, color=gradient_colors['p50'], linestyle='--', linewidth=2.5, label=f'P50={p50:.4f}', alpha=0.8)
        ax1.axvline(p75, color=gradient_colors['p75'], linestyle='--', linewidth=2.5, label=f'P75={p75:.4f}', alpha=0.8)
        
        # 标注映射区间（使用渐变色板）
        y_max = ax1.get_ylim()[1]
        ax1.text(p25, y_max * 0.9, '0.1', ha='center', fontsize=10, fontweight='bold', 
                color=gradient_colors['map_0.1'], bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=gradient_colors['map_0.1']))
        ax1.text((p25 + p50) / 2, y_max * 0.9, '0.4', ha='center', fontsize=10, fontweight='bold', 
                color=gradient_colors['map_0.4'], bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=gradient_colors['map_0.4']))
        ax1.text((p50 + p75) / 2, y_max * 0.9, '0.7', ha='center', fontsize=10, fontweight='bold', 
                color=gradient_colors['map_0.7'], bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=gradient_colors['map_0.7']))
        ax1.text(p75 + (bins[-1] - p75) / 2, y_max * 0.9, '1.0', ha='center', fontsize=10, fontweight='bold', 
                color=gradient_colors['map_1.0'], bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=gradient_colors['map_1.0']))
    else:
        # 极低频APP：显示P90
        p90 = thresholds_info.get('P90', 0)
        ax1.axvline(p90, color=gradient_colors['p90'], linestyle='--', linewidth=2.5, label=f'P90={p90:.4f}', alpha=0.8)
        y_max = ax1.get_ylim()[1]
        ax1.text(0, y_max * 0.9, '0.1', ha='center', fontsize=10, fontweight='bold', 
                color=gradient_colors['map_0.1'], bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=gradient_colors['map_0.1']))
        if p90 > 0:
            ax1.text(p90 / 2, y_max * 0.9, '0.7', ha='center', fontsize=10, fontweight='bold', 
                    color=gradient_colors['map_0.7'], bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=gradient_colors['map_0.7']))
            ax1.text(p90 + (bins[-1] - p90) / 2, y_max * 0.9, '1.0', ha='center', fontsize=10, fontweight='bold', 
                    color=gradient_colors['map_1.0'], bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=gradient_colors['map_1.0']))
    
    ax1.set_xlabel('原始相对偏好 (ratio)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('用户数', fontsize=11, fontweight='bold')
    ax1.set_title(f'{app_label} - 原始分布', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 右子图：映射后的mapped值分布
    ax2 = plt.subplot(3, 4, idx * 2 + 2)
    
    if mapped_col in features_df.columns:
        mapped_data = features_df[mapped_col].value_counts().sort_index()
        # 使用渐变色板：从深蓝到薄荷绿的渐变
        colors_map = {
            0.1: '#2F74B7',  # 深蓝（最低）
            0.4: '#0098CE',  # 青色
            0.7: '#00B9C8',  # 青绿色
            1.0: '#00D5AB'   # 薄荷绿（最高）
        }
        
        bars = ax2.bar([str(v) for v in mapped_data.index], mapped_data.values, 
                       color=[colors_map.get(v, '#0098CE') for v in mapped_data.index],
                       edgecolor='white', linewidth=2, alpha=0.85)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(features_df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('映射值', fontsize=11, fontweight='bold')
        ax2.set_ylabel('用户数', fontsize=11, fontweight='bold')
        ax2.set_title(f'{app_label} - 映射后分布', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(VIZ_DIR / "ratio_binning_mapping.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("    [OK] ratio_binning_mapping.png - 分位数映射可视化")

# ============== Step 11: 生成雷达图 ==============
print("\n[Step 11] 生成客群雷达图...")

apps_order = ['video', 'game', 'social', 'learn', 'shop', 'music']
apps_labels = ['视频', '游戏', '社交', '学习', '电商', '音乐']

def plot_radar(values, labels, title, filename, figsize=(10, 10), color='#2E86AB'):
    """绘制美观的雷达图（统一范围）
    
    参数:
    - values: 数据值数组（原始相对偏好值，范围0-1）
    - labels: 标签数组
    - title: 图表标题
    - filename: 保存文件名
    - figsize: 图表尺寸
    - color: 线条颜色
    """
    import numpy as np
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # 将相对偏好（0-1）转换为百分比（0-100）用于显示
    display_values = values * 100  # 直接转换为0-100的百分比
    
    values_cycle = np.concatenate((display_values, [display_values[0]]))
    angles_cycle = angles + [angles[0]]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    
    # 设置网格样式
    ax.set_theta_offset(np.pi / 2)  # 从顶部开始
    ax.set_theta_direction(-1)  # 顺时针方向
    
    # 固定Y轴范围：0-100（百分比），所有雷达图统一标准
    ax.set_ylim(0, 100)
    
    # 设置径向网格线（固定刻度：0, 20, 40, 60, 80, 100）
    y_ticks = [20, 40, 60, 80, 100]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(t)}%' for t in y_ticks], fontsize=9, color='gray')
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4, color='gray')
    
    # 绘制填充区域
    ax.fill(angles_cycle, values_cycle, alpha=0.3, color=color, edgecolor='none')
    
    # 绘制主线条
    ax.plot(angles_cycle, values_cycle, linewidth=3, color=color, marker='o', 
            markersize=8, markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
    
    # 设置标签样式
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold', color='#333333')
    
    # 添加数值标签（显示百分比）
    for angle, disp_val, label in zip(angles, display_values, labels):
        if disp_val > 2:  # 只显示大于2%的值，避免过于拥挤
            text_label = f'{disp_val:.1f}%'
            offset = 5
            ax.text(angle, disp_val + offset, text_label, 
                   ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
    
    # 设置标题
    ax.set_title(title, fontsize=16, fontweight='bold', pad=30, color='#2C3E50')
    
    # 添加说明
    note = "注：数值为相对偏好百分比（该类APP使用天数占总APP使用天数的比例）"
    fig.text(0.5, 0.02, note, ha='center', fontsize=9, color='#666666', style='italic')
    
    plt.tight_layout()
    plt.savefig(SEG_VIZ_DIR / filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# 单客群雷达图（使用不同颜色方案）
segment_colors = {
    1: '#E74C3C',  # 剧综重度用户 - 红色
    2: '#F39C12',  # 硬核玩家 - 橙色
    3: '#3498DB',  # 社交达人 - 蓝色
    4: '#9B59B6',  # 学习型青年 - 紫色
    5: '#1ABC9C',  # 电商剁手党 - 青色
    6: '#F1C40F'   # 音乐发烧友 - 黄色
}

# 单客群雷达图不再需要计算统一最大值，因为都归一化到0-100
print(f"  [INFO] 雷达图采用归一化显示（0-100），所有客群使用统一Y轴范围")

for seg_id, seg_name in segment_id_map.items():
    prof = segment_profiles.get(seg_name, {})
    vals = []
    for app in apps_order:
        app_pref = prof.get('APP相对偏好', {}).get(app, {})
        vals.append(app_pref.get('均值', 0.0))
    vals = np.array(vals, dtype=float)
    title = f"{seg_name}APP偏好雷达图"
    filename = f"segment_{seg_id}_profile.png"
    color = segment_colors.get(seg_id, '#2E86AB')
    plot_radar(vals, apps_labels, title, filename, figsize=(10, 10), color=color)

# 六客群对比雷达图（归一化版本）
print("  生成对比雷达图 segment_comparison.png...")
num_vars = len(apps_labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles_cycle = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(14, 11), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')

# 设置网格样式
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 固定Y轴范围：0-100
ax.set_ylim(0, 100)

# 设置径向网格线
y_ticks = [20, 40, 60, 80, 100]
ax.set_yticks(y_ticks)
ax.set_yticklabels([str(int(t)) for t in y_ticks], fontsize=9, color='gray')
ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4, color='gray')

# 使用颜色方案
comparison_colors = ['#E74C3C', '#F39C12', '#3498DB', '#9B59B6', '#1ABC9C', '#F1C40F']

for idx, (seg_id, seg_name) in enumerate(segment_id_map.items()):
    prof = segment_profiles.get(seg_name, {})
    vals = []
    for app in apps_order:
        app_pref = prof.get('APP相对偏好', {}).get(app, {})
        vals.append(app_pref.get('均值', 0.0))
    vals = np.array(vals, dtype=float)
    
    # 归一化到0-100
    total = vals.sum()
    if total > 0:
        display_vals = (vals / total) * 100
        # 进一步放大，让形状更清晰
        max_val = display_vals.max()
        if max_val > 0 and max_val < 50:
            scale = 80 / max_val
            display_vals = display_vals * scale
    else:
        display_vals = vals * 100
    
    vals_cycle = np.concatenate((display_vals, [display_vals[0]]))
    
    color = comparison_colors[idx]
    # 绘制填充区域
    ax.fill(angles_cycle, vals_cycle, alpha=0.2, color=color, edgecolor='none')
    # 绘制主线条
    ax.plot(angles_cycle, vals_cycle, linewidth=2.5, color=color, 
            marker='o', markersize=6, markerfacecolor='white', 
            markeredgewidth=1.5, markeredgecolor=color, label=seg_name)

# 设置标签
ax.set_xticks(angles)
ax.set_xticklabels(apps_labels, fontsize=12, fontweight='bold', color='#333333')

# 设置标题
ax.set_title('Z世代6大客群APP偏好对比雷达图（归一化显示）', fontsize=18, fontweight='bold', 
             pad=35, color='#2C3E50')

# 优化图例
legend = ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.05), 
                   fontsize=11, frameon=True, fancybox=True, 
                   shadow=True, framealpha=0.9, edgecolor='gray')
legend.get_frame().set_facecolor('white')

# 添加说明
fig.text(0.5, 0.02, "注：所有客群已归一化到0-100范围显示，便于形状对比", 
         ha='center', fontsize=10, color='#666666', style='italic')

plt.tight_layout()
plt.savefig(SEG_VIZ_DIR / "segment_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("    [OK] segment_1-6_profile.png 与 segment_comparison.png 已生成（归一化显示）")

# ============== Step 12: 新增可视化图表（科研配色） ==============
print("\n[Step 12] 生成新增可视化图表（科研配色）...")

# 科研配色方案
research_colors = {
    'primary': '#2F74B7',      # 深蓝色
    'secondary': '#6FA8EF',    # 浅蓝色
    'light': '#DDF2FF',        # 极浅蓝色
    'accent': '#DCA11D'        # 金黄色
}

# 图1: 阈值可视化（在特征分布上）
print("  生成图1: threshold_visualization.png...")

# 选择关键特征进行可视化
key_features = {
    'video_app_days': {'label': '视频APP使用天数', 'thresholds': ['video_75']},
    'arpu_current': {'label': 'ARPU（元）', 'thresholds': ['arpu_75']},
    'social_app_days': {'label': '社交APP使用天数', 'thresholds': ['social_75']},
    'shop_app_days': {'label': '购物APP使用天数', 'thresholds': ['shop_75']}
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('关键特征分布与阈值可视化', fontsize=16, fontweight='bold', y=0.995)

axes = axes.flatten()
for idx, (feat_col, feat_info) in enumerate(key_features.items()):
    ax = axes[idx]
    
    if feat_col not in features_df.columns:
        ax.text(0.5, 0.5, f'字段 {feat_col} 不存在', 
               ha='center', va='center', transform=ax.transAxes)
        continue
    
    feat_data = features_df[feat_col].dropna()
    
    # 绘制直方图（使用科研配色）
    n, bins, patches = ax.hist(feat_data, bins=50, alpha=0.6, 
                               color=research_colors['light'], 
                               edgecolor=research_colors['primary'], linewidth=1)
    
    # 标注阈值
    for thresh_name in feat_info['thresholds']:
        if thresh_name in thresholds:
            thresh_value = thresholds[thresh_name]
            ax.axvline(thresh_value, color=research_colors['accent'], 
                      linestyle='--', linewidth=2.5, 
                      label=f'{thresh_name}: {thresh_value:.2f}', alpha=0.8)
    
    # 添加分位数线
    p25 = feat_data.quantile(0.25)
    p50 = feat_data.quantile(0.50)
    p75 = feat_data.quantile(0.75)
    
    ax.axvline(p25, color=research_colors['secondary'], linestyle=':', 
              linewidth=2, label=f'P25: {p25:.2f}', alpha=0.7)
    ax.axvline(p50, color=research_colors['primary'], linestyle=':', 
              linewidth=2, label=f'P50: {p50:.2f}', alpha=0.7)
    ax.axvline(p75, color=research_colors['primary'], linestyle='--', 
              linewidth=2, label=f'P75: {p75:.2f}', alpha=0.7)
    
    ax.set_xlabel(feat_info['label'], fontsize=11, fontweight='bold')
    ax.set_ylabel('用户数', fontsize=11, fontweight='bold')
    ax.set_title(feat_info['label'] + '分布', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(VIZ_DIR / "threshold_visualization.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("    [OK] threshold_visualization.png")

print("\n" + "=" * 80)
print("[完成] 软阈值 + 综合得分")
print("=" * 80)
print(f"\n核心逻辑:")
print(f"  [OK] 直接使用cleaned_data.csv原始数据（未标准化）")
print(f"  [OK] 计算相对偏好（ratio）：APP使用天数占比")
print(f"  [OK] 综合打分：相对偏好（主条件）+ 绝对天数/行为（辅助条件）")
print(f"  [OK] 所有用户根据最高得分归入6大客群")
print(f"\n客群分布:")
for segment, count in segment_counts.items():
    pct = count / len(features_df) * 100
    print(f"  {segment}: {count}人 ({pct:.1f}%)")

print(f"\n输出文件:")
print(f"  [DATA] features_with_segments.csv - 特征+标签（供XGBoost使用）")
print(f"  [DATA] cleaned_data_with_labels.csv - 原始数据+标签")
print(f"  [DATA] feature_names.json - 完整特征说明")
print(f"  [DATA] segmentation_thresholds.json - 数据驱动阈值")
print(f"  [DATA] segment_profiles.json - 客群画像")
print(f"  [DATA] segmentation_summary.json - 分群总结")
print(f"  [DATA] segments_data.json - 客群详细数据")
print(f"  [VIZ] segment_distribution_pie.png")
print(f"  [VIZ] app_preference_heatmap.png")
print(f"  [VIZ] ratio_binning_mapping.png（分位数映射可视化）")
print(f"  [VIZ] segment_1-6_profile.png（雷达图）")
print(f"  [VIZ] segment_comparison.png（对比雷达图）")
print(f"  [VIZ] threshold_visualization.png（阈值可视化）")

