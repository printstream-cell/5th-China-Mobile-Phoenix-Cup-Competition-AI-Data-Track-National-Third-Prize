"""
Z世代客群分析 - 特征工程脚本
作者: 算法工程师
日期: 2024-12
功能: 针对6客群设计多维度区分性特征
客群: 剧综重度用户、硬核玩家、社交达人、学习型青年、电商剁手党、音乐发烧友
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============== 配置路径 ==============
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "algorithm" / "outputs" / "data"
MODEL_DIR = BASE_DIR / "algorithm" / "models"

# 创建输出目录
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Z世代客群分析 - 特征工程模块")
print("目标: 针对6客群设计多维度区分性特征")
print("=" * 80)

# ============== Step 1: 加载清洗后数据 ==============
print("\n[Step 1] 加载清洗后数据...")

df = pd.read_csv(DATA_DIR / "cleaned_data.csv")

# 转换所有APP使用字段和流量字段为数值型
numeric_cols = [col for col in df.columns if any(keyword in col for keyword in [
    'APP_USE', 'APP', 'flux', 'dou', 'pro', 'mou', 'ARPU', 'FEE', 'BAL', 'DURA', 'INNET'
])]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"[OK] 数据加载完成: {len(df):,} 条样本, {df.shape[1]} 个字段")

# ============== Step 2: 特征提取 ==============
print("\n[Step 2] 特征提取...")
print("\n特征设计原则: 针对6个客群的核心区分维度")
print("  1. 剧综重度用户 → 视频APP高使用")
print("  2. 硬核玩家 → 游戏APP高使用 + 深夜活跃")
print("  3. 社交达人 → 社交APP高使用 + 通话频繁")
print("  4. 学习型青年 → 学习/阅读/办公APP使用")
print("  5. 电商剁手党 → 购物/餐饮APP使用")
print("  6. 音乐发烧友 → 音乐APP高使用")

# 创建特征DataFrame
features = pd.DataFrame()

# ============== A. 基础属性特征 (5个) ==============
print("\n[A] 基础属性特征 (5个)...")

# A1. 年龄
features['age'] = df['AGE']

# A2. 在网时长(月)
features['tenure_months'] = df['INNET_DURA'] / 30

# A3. 城市级别
features['city_tier'] = df['city_tier'] if 'city_tier' in df.columns else 3

# A4. 是否5G用户
features['is_5g_user'] = df['IS_ORD_5G_PACKAGE'] if 'IS_ORD_5G_PACKAGE' in df.columns else 0

# A5. 设备价值
features['device_value'] = df['device_brand_value'] if 'device_brand_value' in df.columns else 2

print(f"  [OK] 基础属性特征: {5} 个")

# ============== B. 消费行为特征 (5个) ==============
print("\n[B] 消费行为特征 (5个)...")

# B1. 当前ARPU
features['arpu_current'] = df['DIS_ARPU']

# B2. ARPU趋势
if 'N3M_AVG_DIS_ARPU' in df.columns:
    features['arpu_trend'] = (df['DIS_ARPU'] - df['N3M_AVG_DIS_ARPU']) / (df['N3M_AVG_DIS_ARPU'] + 1)
else:
    features['arpu_trend'] = 0

# B3. 套餐费
features['package_fee'] = df['PRI_PACKAGE_FEE'] if 'PRI_PACKAGE_FEE' in df.columns else 0

# B4. 账户余额
features['account_balance'] = df['ACCT_BAL'] if 'ACCT_BAL' in df.columns else 0

# B5. 是否双卡
features['is_dual_sim'] = df['IS_DUALSIM_USER'] if 'IS_DUALSIM_USER' in df.columns else 0

print(f"  [OK] 消费行为特征: {5} 个")

# ============== C. APP使用特征 (核心区分特征) ==============
print("\n[C] APP使用特征 - 核心区分特征...")

# C1-C3. 视频类 (剧综重度用户核心特征)
features['video_app_days'] = df['N3M_AVG_VIDEO_APP_USE_DAYS'] if 'N3M_AVG_VIDEO_APP_USE_DAYS' in df.columns else 0
features['video_app_duration'] = df['N3M_AVG_VIDEO_APP_USE_DURA'] if 'N3M_AVG_VIDEO_APP_USE_DURA' in df.columns else 0
features['video_intensity'] = features['video_app_duration'] / (features['video_app_days'] + 1)

print(f"  [OK] 视频类特征: 天数均值={features['video_app_days'].mean():.1f}, 时长均值={features['video_app_duration'].mean():.1f}分")

# C4-C6. 游戏类 (硬核玩家核心特征)
features['game_app_days'] = df['N3M_AVG_GAME_APP_USE_DAYS'] if 'N3M_AVG_GAME_APP_USE_DAYS' in df.columns else 0
features['game_app_duration'] = df['N3M_AVG_GAME_APP_USE_DURA'] if 'N3M_AVG_GAME_APP_USE_DURA' in df.columns else 0
features['game_intensity'] = features['game_app_duration'] / (features['game_app_days'] + 1)

print(f"  [OK] 游戏类特征: 天数均值={features['game_app_days'].mean():.1f}, 时长均值={features['game_app_duration'].mean():.1f}分")

# C7-C9. 社交类 (社交达人核心特征)
features['social_app_days'] = df['N3M_AVG_SOCIAL_APP_USE_DAYS'] if 'N3M_AVG_SOCIAL_APP_USE_DAYS' in df.columns else 0
features['social_app_cnt'] = df['N3M_AVG_SOCIAL_APP_USE_CNT'] if 'N3M_AVG_SOCIAL_APP_USE_CNT' in df.columns else 0
features['social_intensity'] = features['social_app_cnt'] / (features['social_app_days'] + 1)

print(f"  [OK] 社交类特征: 天数均值={features['social_app_days'].mean():.1f}, 次数均值={features['social_app_cnt'].mean():.1f}")

# C10-C12. 学习类 (学习型青年核心特征)
features['learn_app_days'] = df['N3M_AVG_LEARN_APP_USE_DAYS'] if 'N3M_AVG_LEARN_APP_USE_DAYS' in df.columns else 0
features['read_app_days'] = df['N3M_AVG_READ_APP_USE_DAYS'] if 'N3M_AVG_READ_APP_USE_DAYS' in df.columns else 0
features['office_app_days'] = df['N3M_AVG_OFFICE_APP_USE_DAYS'] if 'N3M_AVG_OFFICE_APP_USE_DAYS' in df.columns else 0

print(f"  [OK] 学习类特征: 学习={features['learn_app_days'].mean():.1f}, 阅读={features['read_app_days'].mean():.1f}, 办公={features['office_app_days'].mean():.1f}天")

# C13-C15. 购物类 (电商剁手党核心特征)
features['shop_app_days'] = df['N3M_AVG_SHOP_APP_USE_DAYS'] if 'N3M_AVG_SHOP_APP_USE_DAYS' in df.columns else 0
features['restaurant_app_days'] = df['N3M_AVG_RESTAURANT_APP_USE_DAYS'] if 'N3M_AVG_RESTAURANT_APP_USE_DAYS' in df.columns else 0
features['consumption_intensity'] = features['shop_app_days'] + features['restaurant_app_days']

print(f"  [OK] 购物类特征: 购物={features['shop_app_days'].mean():.1f}, 餐饮={features['restaurant_app_days'].mean():.1f}天")

# C16-C18. 音乐类 (音乐发烧友核心特征)
features['music_app_days'] = df['N3M_AVG_MUSIC_APP_USE_DAYS'] if 'N3M_AVG_MUSIC_APP_USE_DAYS' in df.columns else 0
features['music_app_duration'] = df['N3M_AVG_MUSIC_APP_USE_DURA'] if 'N3M_AVG_MUSIC_APP_USE_DURA' in df.columns else 0
features['music_intensity'] = features['music_app_duration'] / (features['music_app_days'] + 1)

print(f"  [OK] 音乐类特征: 天数均值={features['music_app_days'].mean():.1f}, 时长均值={features['music_app_duration'].mean():.1f}分")

# C19-C21. 直播类 (剧综重度用户 / 硬核玩家辅助特征)
features['live_app_days'] = df['N3M_AVG_LIVE_BROAD_APP_USE_DAYS'] if 'N3M_AVG_LIVE_BROAD_APP_USE_DAYS' in df.columns else 0
features['live_app_duration'] = df['N3M_AVG_LIVE_BROAD_APP_USE_DURA'] if 'N3M_AVG_LIVE_BROAD_APP_USE_DURA' in df.columns else 0
features['live_intensity'] = features['live_app_duration'] / (features['live_app_days'] + 1)

print(f"  [OK] 直播类特征: 天数均值={features['live_app_days'].mean():.1f}, 时长均值={features['live_app_duration'].mean():.1f}分")

# C22. 出行APP (旅行APP使用, 反映消费能力与生活方式)
features['trip_app_days'] = df['N3M_AVG_TRIP_APP_USE_DAYS'] if 'N3M_AVG_TRIP_APP_USE_DAYS' in df.columns else 0

# C23. 邮件APP (学习/办公场景, 学习型青年辅助特征)
features['email_app_days'] = df['N3M_AVG_EMAIL_APP_USE_DAYS'] if 'N3M_AVG_EMAIL_APP_USE_DAYS' in df.columns else 0

print(f"  [OK] 出行/邮件类特征: 旅行={features['trip_app_days'].mean():.1f}, 邮件={features['email_app_days'].mean():.1f}天")

print(f"\n  [OK] APP使用特征数量: {len([c for c in features.columns if 'app_' in c or '_app_' in c])} 个")

# ============== D. 流量行为特征 (5个) ==============
print("\n[D] 流量行为特征 (5个)...")

# D1. 白天流量占比
if 'day_flux' in df.columns and 'night_flux' in df.columns:
    total_flux = df['day_flux'] + df['night_flux'] + 1
    features['day_flux_ratio'] = df['day_flux'] / total_flux
    
    # D2. 深夜活跃度(硬核玩家辅助特征)
    features['night_owl_score'] = df['night_flux'] / total_flux
else:
    features['day_flux_ratio'] = 0.5
    features['night_owl_score'] = 0.5

# D3. 周末流量占比
features['weekend_ratio'] = df['week_pro'] if 'week_pro' in df.columns else 0.3

# D4. 总在线天数
features['total_online_days'] = df['dou'] if 'dou' in df.columns else 0

# D5. 日均流量
if 'day_flux' in df.columns and 'night_flux' in df.columns:
    features['daily_flux_avg'] = (df['day_flux'] + df['night_flux']) / (features['total_online_days'] + 1)
else:
    features['daily_flux_avg'] = 0

print(f"  [OK] 流量行为特征: {5} 个")
print(f"    - 深夜活跃度均值: {features['night_owl_score'].mean():.2f}")
print(f"    - 周末占比均值: {features['weekend_ratio'].mean():.2f}")

# ============== E. 场景 / 社交结构 / 金融APP补充特征 ==============
print("\n[E] 场景 / 社交结构 / 金融APP补充特征...")

# E1. 学校/公司驻留特征 (学习型青年 / 职场青年场景)
features['school_resident_days'] = df['T_school_resident'] if 'T_school_resident' in df.columns else 0
features['school_night_resident_days'] = df['T_school_night_resident'] if 'T_school_night_resident' in df.columns else 0
features['company_resident_days'] = df['T_company_resident'] if 'T_company_resident' in df.columns else 0

# E2. 通话社交广度特征 (社交达人辅助特征)
features['unique_call_users'] = df['L3M_AVG_DIFF_CALL_USER_CNT'] if 'L3M_AVG_DIFF_CALL_USER_CNT' in df.columns else 0
features['circle_size'] = df['circle_num'] if 'circle_num' in df.columns else 0

# E3. 金融APP使用特征 (电商剁手党消费能力核心补充)
features['finance_app_days'] = df['N3M_AVG_FINANC_APP_USE_DAYS'] if 'N3M_AVG_FINANC_APP_USE_DAYS' in df.columns else 0
features['bank_app_days'] = df['N3M_AVG_BNK_APP_USE_DAYS'] if 'N3M_AVG_BNK_APP_USE_DAYS' in df.columns else 0
features['account_app_days'] = df['N3M_AVG_ACCNT_APP_USE_DAYS'] if 'N3M_AVG_ACCNT_APP_USE_DAYS' in df.columns else 0

print(f"  [OK] 学校/公司驻留特征已构建")
print(f"  [OK] 通话社交广度特征已构建")
print(f"  [OK] 金融APP使用特征已构建")

# ============== F. 综合特征 (2个) ==============
print("\n[F] 综合特征 (2个)...")

# F1. APP多样性(使用APP类型数)
app_day_cols = [
    'video_app_days', 'game_app_days', 'social_app_days',
    'learn_app_days', 'read_app_days', 'office_app_days',
    'shop_app_days', 'restaurant_app_days', 'music_app_days',
    'live_app_days', 'trip_app_days', 'email_app_days'
]
features['app_diversity'] = features[app_day_cols].apply(lambda x: (x > 0).sum(), axis=1)

# F2. 主导APP类型(最常用的APP类型)
def get_dominant_app(row):
    try:
        app_usage = {
            'video': float(row['video_app_days']),
            'game': float(row['game_app_days']),
            'social': float(row['social_app_days']),
            'learn': float(row['learn_app_days'] + row['read_app_days'] + row['office_app_days']),
            'shop': float(row['shop_app_days'] + row['restaurant_app_days']),
            'music': float(row['music_app_days']),
            'live': float(row['live_app_days']),
            'trip': float(row['trip_app_days']),
            'email': float(row['email_app_days'])
        }
        max_val = max(app_usage.values())
        return max(app_usage, key=app_usage.get) if max_val > 0 else 'none'
    except:
        return 'none'

features['dominant_app_type'] = features[app_day_cols + ['learn_app_days', 'read_app_days', 'office_app_days']].apply(get_dominant_app, axis=1)

# 主导APP类型编码
dominant_app_map = {
    'video': 0,
    'game': 1,
    'social': 2,
    'learn': 3,
    'shop': 4,
    'music': 5,
    'live': 6,
    'trip': 7,
    'email': 8,
    'none': 9
}
features['dominant_app_encoded'] = features['dominant_app_type'].map(dominant_app_map)

print(f"  [OK] 综合特征: {2} 个")
print(f"    - APP多样性均值: {features['app_diversity'].mean():.1f} 种")

# ============== Step 3: 特征汇总 ==============
print("\n[Step 3] 特征汇总...")

# 选择数值型特征(用于聚类和分类)
numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()

# 移除非特征列
exclude_cols = ['dominant_app_type']  # 仅保留编码后的版本
numeric_features = [col for col in numeric_features if col not in exclude_cols]

X = features[numeric_features].copy()

print(f"  [OK] 特征总数: {len(numeric_features)} 个")
print(f"  [OK] 样本数: {len(X):,} 条")

# 特征统计
print(f"\n  特征统计:")
print(f"    - A类(基础属性): 5 个")
print(f"    - B类(消费行为): 5 个")
print(f"    - C类(APP使用): 包含视频/游戏/社交/学习/购物/音乐/直播/出行/邮件等多维特征")
print(f"    - D类(流量行为): 5 个")
print(f"    - E类(场景/社交结构/金融APP): 8 个")
print(f"    - F类(综合特征): 2 个")

# ============== Step 4: 特征标准化 ==============
print("\n[Step 4] 特征标准化...")

# 使用StandardScaler进行Z-score标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_features)

print(f"  [OK] 标准化完成: {X_scaled.shape}")
print(f"    - 均值: {X_scaled.mean():.6f}")
print(f"    - 标准差: {X_scaled.std():.6f}")

# ============== Step 5: 保存特征数据 ==============
print("\n[Step 5] 保存特征数据...")

# 5.1 保存特征矩阵
X_scaled_df['USER_ID'] = df['USER_ID'].values if 'USER_ID' in df.columns else range(len(X_scaled_df))
output_file = DATA_DIR / "features_data.csv"
X_scaled_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"  [OK] 特征矩阵已保存: {output_file}")
print(f"    - 文件大小: {output_file.stat().st_size / 1024**2:.2f} MB")

# 5.2 保存Scaler模型
scaler_file = MODEL_DIR / "feature_scaler.pkl"
joblib.dump(scaler, scaler_file)
print(f"  [OK] Scaler已保存: {scaler_file}")

# 5.3 保存特征名称列表
feature_names_file = DATA_DIR / "feature_names.json"
feature_info = {
    "feature_names": numeric_features,
    "num_features": len(numeric_features),
        "feature_categories": {
        "基础属性": ["age", "tenure_months", "city_tier", "is_5g_user", "device_value"],
        "消费行为": ["arpu_current", "arpu_trend", "package_fee", "account_balance", "is_dual_sim"],
        "APP使用_视频": ["video_app_days", "video_app_duration", "video_intensity"],
        "APP使用_游戏": ["game_app_days", "game_app_duration", "game_intensity"],
        "APP使用_社交": ["social_app_days", "social_app_cnt", "social_intensity"],
        "APP使用_学习": ["learn_app_days", "read_app_days", "office_app_days", "email_app_days"],
        "APP使用_购物": ["shop_app_days", "restaurant_app_days", "consumption_intensity"],
        "APP使用_音乐": ["music_app_days", "music_app_duration", "music_intensity"],
        "APP使用_直播/出行": ["live_app_days", "live_app_duration", "live_intensity", "trip_app_days"],
        "流量行为": ["day_flux_ratio", "night_owl_score", "weekend_ratio", "total_online_days", "daily_flux_avg"],
        "场景/社交结构/金融": [
            "school_resident_days",
            "school_night_resident_days",
            "company_resident_days",
            "unique_call_users",
            "circle_size",
            "finance_app_days",
            "bank_app_days",
            "account_app_days"
        ],
        "综合特征": ["app_diversity", "dominant_app_encoded"]
    }
}

with open(feature_names_file, 'w', encoding='utf-8') as f:
    json.dump(feature_info, f, ensure_ascii=False, indent=2)
print(f"  [OK] 特征名称已保存: {feature_names_file}")

# ============== Step 6: 特征重要性预分析 ==============
print("\n[Step 6] 特征重要性预分析...")

# 计算各特征的方差(标准化后)
feature_variance = pd.DataFrame({
    '特征名': numeric_features,
    '方差': X_scaled.var(axis=0)
}).sort_values('方差', ascending=False)

print(f"\n  TOP10高方差特征(区分能力强):")
print(feature_variance.head(10).to_string(index=False))

# ============== Step 7: 6客群核心特征验证 ==============
print("\n[Step 7] 6客群核心特征验证...")

# 计算各客群核心特征的非零比例
segment_features = {
    "剧综重度用户": "video_app_days",
    "硬核玩家": "game_app_days",
    "社交达人": "social_app_days",
    "学习型青年": "learn_app_days",
    "电商剁手党": "shop_app_days",
    "音乐发烧友": "music_app_days"
}

print(f"\n  各客群核心特征使用率:")
for segment, feature in segment_features.items():
    if feature in features.columns:
        usage_rate = (features[feature] > 0).sum() / len(features) * 100
        avg_days = features[features[feature] > 0][feature].mean()
        print(f"    {segment:12s}: {usage_rate:5.1f}% 用户使用, 平均 {avg_days:5.1f} 天/月")

# ============== Step 8: 生成特征工程报告 ==============
print("\n[Step 8] 生成特征工程报告...")

feature_report = {
    "特征总数": len(numeric_features),
    "样本数": len(X),
    "特征分类统计": {
        "基础属性": 5,
        "消费行为": 5,
        "APP使用": 23,
        "流量行为": 5,
        "场景/社交结构/金融": 8,
        "综合特征": 2
    },
    "6客群核心特征": {
        "剧综重度用户": {
            "核心特征": ["video_app_days", "video_app_duration", "video_intensity", "live_app_days", "live_app_duration", "live_intensity"],
            "使用率": f"{(features['video_app_days'] > 0).sum() / len(features) * 100:.1f}%"
        },
        "硬核玩家": {
            "核心特征": ["game_app_days", "game_app_duration", "game_intensity", "night_owl_score", "live_app_days"],
            "使用率": f"{(features['game_app_days'] > 0).sum() / len(features) * 100:.1f}%"
        },
        "社交达人": {
            "核心特征": ["social_app_days", "social_app_cnt", "social_intensity", "unique_call_users", "circle_size"],
            "使用率": f"{(features['social_app_days'] > 0).sum() / len(features) * 100:.1f}%"
        },
        "学习型青年": {
            "核心特征": ["learn_app_days", "read_app_days", "office_app_days", "email_app_days", "school_resident_days", "school_night_resident_days"],
            "使用率": f"{((features['learn_app_days'] > 0) | (features['read_app_days'] > 0)).sum() / len(features) * 100:.1f}%"
        },
        "电商剁手党": {
            "核心特征": ["shop_app_days", "restaurant_app_days", "consumption_intensity", "finance_app_days", "bank_app_days", "account_app_days"],
            "使用率": f"{(features['shop_app_days'] > 0).sum() / len(features) * 100:.1f}%"
        },
        "音乐发烧友": {
            "核心特征": ["music_app_days", "music_app_duration", "music_intensity"],
            "使用率": f"{(features['music_app_days'] > 0).sum() / len(features) * 100:.1f}%"
        }
    },
    "特征质量": {
        "标准化方法": "StandardScaler (Z-score)",
        "缺失值": 0,
        "标准化后均值": float(X_scaled.mean()),
        "标准化后标准差": float(X_scaled.std())
    }
}

report_file = DATA_DIR / "feature_engineering_report.json"
with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(feature_report, f, ensure_ascii=False, indent=2)
print(f"  [OK] 特征工程报告已保存: {report_file}")

print("\n" + "=" * 80)
print("[OK] 特征工程完成!")
print("=" * 80)
print(f"\n交付物:")
print(f"  1. 特征矩阵: {output_file}")
print(f"  2. Scaler模型: {scaler_file}")
print(f"  3. 特征名称: {feature_names_file}")
print(f"  4. 工程报告: {report_file}")
print(f"\n下一步: 运行 03_kmeans_clustering.py 进行K-Means聚类")

