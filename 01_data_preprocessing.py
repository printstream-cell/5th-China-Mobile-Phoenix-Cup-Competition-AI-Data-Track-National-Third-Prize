"""
Z世代客群分析 - 数据预处理脚本

"""

import pandas as pd
import numpy as np
import os
import sys
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
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "algorithm" / "outputs" / "data"
VIZ_DIR = BASE_DIR / "algorithm" / "outputs" / "visualizations" / "eda"

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Z世代客群分析 - 数据预处理模块")
print("=" * 80)

# ============== Step 1: 数据加载 ==============
print("\n[Step 1] 数据加载中...")

# 读取原始数据 - 自动尝试多种编码
data_file = DATA_DIR / "data_fixed.csv"
encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']

df_raw = None
for encoding in encodings:
    try:
        df_raw = pd.read_csv(data_file, encoding=encoding)
        print(f"  成功使用编码: {encoding}")
        break
    except (UnicodeDecodeError, Exception) as e:
        continue

if df_raw is None:
    print(f"[ERROR] 错误: 无法读取数据文件，尝试了所有编码方式")
    sys.exit(1)

print(f"[OK] 数据加载完成")
print(f"  - 总样本数: {len(df_raw):,} 条")
print(f"  - 字段数量: {df_raw.shape[1]} 个")
print(f"  - 内存占用: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============== Step 2: 数据概览 ==============
print("\n[Step 2] 数据概览统计...")

# 2.1 数据类型分布
dtype_counts = df_raw.dtypes.value_counts()
print(f"\n数据类型分布:")
for dtype, count in dtype_counts.items():
    print(f"  - {dtype}: {count} 个字段")

# 2.2 缺失值统计
missing_stats = pd.DataFrame({
    '字段名': df_raw.columns,
    '缺失数量': df_raw.isnull().sum(),
    '缺失率': (df_raw.isnull().sum() / len(df_raw) * 100).round(2)
})
missing_stats = missing_stats[missing_stats['缺失数量'] > 0].sort_values('缺失率', ascending=False)

print(f"\n缺失值统计 (TOP10):")
if len(missing_stats) > 0:
    print(missing_stats.head(10).to_string(index=False))
else:
    print("  [OK] 无缺失值")

# 2.3 年龄分布检查
print(f"\n年龄分布:")
print(f"  - 最小年龄: {df_raw['AGE'].min()}")
print(f"  - 最大年龄: {df_raw['AGE'].max()}")
print(f"  - 平均年龄: {df_raw['AGE'].mean():.1f}")
print(f"  - Z世代(15-30岁,1995-2010年生)占比: {((df_raw['AGE'] >= 15) & (df_raw['AGE'] <= 30)).sum() / len(df_raw) * 100:.1f}%")

# ============== Step 2.5: 生成原始数据缺失值报告 ==============
print("\n[Step 2.5] 生成原始数据缺失值可视化...")

# 创建可视化输出目录
eda_dir = OUTPUT_DIR.parent / "visualizations" / "eda"
eda_dir.mkdir(parents=True, exist_ok=True)

# 缺失值分布图（基于原始数据）
print("  生成图表: missing_values_distribution.png (原始数据)...")
missing_df_raw = pd.DataFrame({
    '字段': df_raw.columns,
    '缺失数': df_raw.isnull().sum(),
    '缺失率': df_raw.isnull().sum() / len(df_raw) * 100
})
missing_df_raw = missing_df_raw[missing_df_raw['缺失数'] > 0].sort_values('缺失率', ascending=False).head(30)

if len(missing_df_raw) > 0:
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(range(len(missing_df_raw)), missing_df_raw['缺失率'].values, color='coral')
    ax.set_yticks(range(len(missing_df_raw)))
    ax.set_yticklabels(missing_df_raw['字段'].values, fontsize=9)
    ax.set_xlabel('缺失率 (%)', fontsize=12)
    ax.set_title('TOP30字段缺失值分布 (原始数据)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(eda_dir / "missing_values_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [OK] 发现{len(missing_df_raw)}个字段有缺失值")
else:
    print("    [OK] 原始数据无缺失值")

# ============== Step 3: 数据清洗 ==============
print("\n[Step 3] 数据清洗...")

df = df_raw.copy()
initial_count = len(df)

# 3.0 先转换关键字段类型
# 转换数值型字段(可能为字符串，需要移除逗号)
numeric_fields = ['DIS_ARPU', 'N3M_AVG_DIS_ARPU', 'PRI_PACKAGE_FEE', 'ACCT_BAL', 
                  'cm_total_mou', 'day_flux', 'night_flux', 'INNET_DURA', 'AGE']
for field in numeric_fields:
    if field in df.columns:
        # 移除字符串中的逗号，然后转换为数值
        if df[field].dtype == 'object':
            df[field] = df[field].astype(str).str.replace(',', '').replace('nan', '')
        df[field] = pd.to_numeric(df[field], errors='coerce')

print(f"  [OK] 数据类型转换完成")

# 3.1 筛选Z世代用户(1995-2010年出生，即15-30岁)
# 赛题明确要求：对Z时代人群进行画像分析
# 数据采集时间假设：2024年
# 年龄定义：15岁(2010年生) - 30岁(1995年生)
# 必须在EDA之前筛选，确保所有分析都基于Z世代数据
age_before = len(df)
df = df[(df['AGE'] >= 15) & (df['AGE'] <= 30)]
print(f"  [OK] 筛选Z世代用户(15-30岁): {len(df):,} 条")
print(f"      (删除非Z世代用户: {age_before - len(df):,} 条)")

# 3.2 删除极端异常值(使用IQR方法，保留99%以上数据)
# 3.2.1 ARPU值异常 - 使用IQR方法
if 'DIS_ARPU' in df.columns:
    Q1 = df['DIS_ARPU'].quantile(0.01)
    Q3 = df['DIS_ARPU'].quantile(0.99)
    arpu_before = len(df)
    df = df[(df['DIS_ARPU'] >= Q1) & (df['DIS_ARPU'] <= Q3)]
    print(f"  [OK] ARPU值异常检测(IQR): 删除 {arpu_before - len(df)} 条极端异常")

print(f"\n  清洗后样本数: {len(df):,} 条 (保留率: {len(df)/initial_count*100:.1f}%)")

# 3.3 重置索引
df.reset_index(drop=True, inplace=True)

# ============== Step 4: 缺失值处理 ==============
print("\n[Step 4] 缺失值处理...")

# 4.1 数值型字段: 0填充(APP使用类字段) 或 中位数填充(其他)
app_use_cols = [col for col in df.columns if 'APP_USE' in col]
numeric_cols = df.select_dtypes(include=[np.number]).columns

# APP使用字段用0填充(表示未使用)
for col in app_use_cols:
    if col in df.columns:
        df[col].fillna(0, inplace=True)

# 其他数值字段用中位数填充
other_numeric = [col for col in numeric_cols if col not in app_use_cols]
for col in other_numeric:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

print(f"  [OK] APP使用字段(0填充): {len(app_use_cols)} 个")
print(f"  [OK] 其他数值字段(中位数填充): {len(other_numeric)} 个")

# 4.2 类别型字段: "未知"填充
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna('未知', inplace=True)

print(f"  [OK] 类别字段(未知填充): {len(categorical_cols)} 个")

# 验证无缺失
remaining_missing = df.isnull().sum().sum()
print(f"\n  最终缺失值数量: {remaining_missing}")

# ============== Step 5: 数据类型转换与编码 ==============
print("\n[Step 5] 数据类型转换与编码...")

# 5.1 是/否字段 → 0/1编码
yes_no_cols = [
    'IS_ORD_5G_PACKAGE', 'IS_ORD_FAM_BUSI', 'IS_DUALSIM_USER',
    'IS_SINGLE_PHN_CUST', 'IS_OFFSET_KEEPNO', 'IS_SELFNET_BRD_USER',
    'IS_DIFF_BRD_USER', 'IS_TERM_CONTR_USER', 'IS_CONTR_BIND_USER'
]

for col in yes_no_cols:
    if col in df.columns:
        df[col] = df[col].map({'是': 1, '否': 0, '未知': 0}).fillna(0).astype(int)

print(f"  [OK] 是/否字段二值编码: {len([c for c in yes_no_cols if c in df.columns])} 个")

# 5.2 终端品牌编码(按品牌价值)
brand_value_map = {
    '华为': 3,
    '苹果': 4,
    'OPPO': 2,
    'vivo': 2,
    '小米科技': 2,
    'realme': 2,
    '荣耀': 3,
    '三星': 3,
    '欧珀': 2,
    '北京小米科技有限责任公司': 2
}

if 'TERM_BRAND' in df.columns:
    df['device_brand_value'] = df['TERM_BRAND'].map(brand_value_map).fillna(1).astype(int)
    print(f"  [OK] 终端品牌编码: {df['device_brand_value'].nunique()} 个等级")

# 5.3 城市级别编码
city_tier_map = {
    '呼和浩特': 2, '包头': 2, '鄂尔多斯': 3, '赤峰': 3, '通辽': 3,
    '呼伦贝尔': 3, '巴彦淖尔': 4, '乌兰察布': 4, '兴安盟': 4,
    '锡林郭勒盟': 4, '阿拉善盟': 5, '乌海': 4, '鄂温克族自治旗': 4
}

if 'CITY' in df.columns:
    df['city_tier'] = df['CITY'].map(city_tier_map).fillna(4).astype(int)
    print(f"  [OK] 城市级别编码: {df['city_tier'].nunique()} 个等级")

# 5.4 区域类型(城市/农村)
if 'AREA_ID' in df.columns:
    df['is_urban'] = df['AREA_ID'].apply(lambda x: 0 if '农村' in str(x) else 1)
    print(f"  [OK] 区域类型编码: 城市({df['is_urban'].sum()}) / 农村({(df['is_urban']==0).sum()})")

# ============== Step 6: 保存清洗后数据 ==============
print("\n[Step 6] 保存清洗后数据...")

output_file = OUTPUT_DIR / "cleaned_data.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"  [OK] 已保存: {output_file}")
print(f"  - 清洗后样本数: {len(df):,}")
print(f"  - 清洗后字段数: {df.shape[1]}")
print(f"  - 文件大小: {output_file.stat().st_size / 1024**2:.2f} MB")

# ============== Step 7: 生成清洗报告 ==============
print("\n[Step 7] 生成数据清洗报告...")

cleaning_report = {
    "原始数据": {
        "样本数": int(initial_count),
        "字段数": int(df_raw.shape[1])
    },
    "清洗后数据": {
        "样本数": int(len(df)),
        "字段数": int(df.shape[1]),
        "保留率": f"{len(df)/initial_count*100:.2f}%"
    },
    "数据质量": {
        "缺失值数量": int(remaining_missing),
        "重复行数量": int(df.duplicated().sum()),
        "Z世代占比": f"{len(df)/initial_count*100:.1f}%"
    },
    "年龄统计": {
        "最小年龄": int(df['AGE'].min()),
        "最大年龄": int(df['AGE'].max()),
        "平均年龄": float(df['AGE'].mean()),
        "中位数年龄": float(df['AGE'].median())
    },
    "处理步骤": {
        "步骤1": "筛选Z世代用户(15-30岁,1995-2010年生,数据采集时间2024年)",
        "步骤2": "删除ARPU极端异常值(IQR方法,保留99%数据)",
        "步骤3": "缺失值填充(APP使用0填充,数值中位数填充,类别未知填充)",
        "步骤4": "是/否字段二值编码(0/1)",
        "步骤5": "终端品牌价值编码(1-4等级)",
        "步骤6": "城市级别编码(2-5等级)",
        "步骤7": "区域类型编码(城市/农村)"
    }
}

report_file = OUTPUT_DIR / "cleaning_report.json"
with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(cleaning_report, f, ensure_ascii=False, indent=2)

print(f"  [OK] 清洗报告已保存: {report_file}")

# ============== Step 8: 数据概览统计 ==============
print("\n[Step 8] 清洗后数据概览...")

print(f"\n关键字段统计:")
print(f"  年龄: {df['AGE'].min()}-{df['AGE'].max()} 岁 (均值 {df['AGE'].mean():.1f})")
print(f"  ARPU: {df['DIS_ARPU'].min():.1f}-{df['DIS_ARPU'].max():.1f} 元 (均值 {df['DIS_ARPU'].mean():.1f})")
print(f"  在网时长: {df['INNET_DURA'].min()}-{df['INNET_DURA'].max()} 天 (均值 {df['INNET_DURA'].mean():.0f})")

if 'IS_ORD_5G_PACKAGE' in df.columns:
    print(f"  5G用户占比: {df['IS_ORD_5G_PACKAGE'].mean()*100:.1f}%")

if 'IS_DUALSIM_USER' in df.columns:
    print(f"  双卡用户占比: {df['IS_DUALSIM_USER'].mean()*100:.1f}%")

# APP使用统计
app_stats = {}
app_types = ['VIDEO', 'GAME', 'SOCIAL', 'MUSIC', 'SHOP', 'LEARN']
for app_type in app_types:
    col_name = f'N3M_AVG_{app_type}_APP_USE_DAYS'
    if col_name in df.columns:
        app_stats[app_type] = df[col_name].mean()

print(f"\nAPP使用天数统计(月均):")
for app, days in sorted(app_stats.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {app}: {days:.1f} 天")

# ============== Step 9: 生成EDA可视化报告 ==============
print("\n[Step 9] 生成探索性数据分析(EDA)报告（清洗后数据）...")

# 确保输出目录存在
eda_dir.mkdir(parents=True, exist_ok=True)

# 9.1 年龄分布直方图
print("  生成图表1: age_distribution.png...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['AGE'].dropna(), bins=16, color='skyblue', edgecolor='black')
ax.axvline(x=15, color='red', linestyle='--', linewidth=2, label='Z世代下限(15岁,2010年生)')
ax.axvline(x=30, color='red', linestyle='--', linewidth=2, label='Z世代上限(30岁,1995年生)')
ax.set_xlabel('年龄', fontsize=12)
ax.set_ylabel('用户数量', fontsize=12)
ax.set_title('Z世代用户年龄分布 (15-30岁, 1995-2010年出生)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(eda_dir / "age_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("    [OK] age_distribution.png")

# 9.2 ARPU分布直方图
print("  生成图表2: arpu_distribution.png...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['DIS_ARPU'].dropna(), bins=50, color='lightgreen', edgecolor='black')
ax.set_xlabel('ARPU (元)', fontsize=12)
ax.set_ylabel('用户数量', fontsize=12)
ax.set_title('用户ARPU分布', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(eda_dir / "arpu_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("    [OK] arpu_distribution.png")

# 9.4 APP使用天数分布(多子图)
print("  生成图表3: app_usage_distribution.png...")
app_types = ['VIDEO', 'GAME', 'SOCIAL', 'MUSIC', 'SHOP', 'LEARN']
app_cols = [f'N3M_AVG_{app}_APP_USE_DAYS' for app in app_types]
existing_app_cols = [col for col in app_cols if col in df.columns]

if len(existing_app_cols) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(existing_app_cols):
        if idx < 6:
            app_name = col.split('_')[2]
            axes[idx].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(f'{app_name} APP使用天数', fontsize=10)
            axes[idx].set_ylabel('用户数量', fontsize=10)
            axes[idx].set_title(f'{app_name} APP使用分布', fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(eda_dir / "app_usage_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    [OK] app_usage_distribution.png")

# 9.4 异常值箱线图(关键特征)
print("  生成图表4: outliers_boxplot.png...")
# 准备关键特征数据，对流量字段转换单位为GB
key_features_data = {}
feature_labels = {}

if 'DIS_ARPU' in df.columns:
    key_features_data['DIS_ARPU'] = df['DIS_ARPU'].dropna()
    feature_labels['DIS_ARPU'] = 'DIS_ARPU (元)'

if 'day_flux' in df.columns:
    # 转换为GB (1GB = 1,073,741,824 bytes)
    key_features_data['day_flux'] = (df['day_flux'].dropna() / 1073741824)
    feature_labels['day_flux'] = 'day_flux (GB)'

if 'night_flux' in df.columns:
    key_features_data['night_flux'] = (df['night_flux'].dropna() / 1073741824)
    feature_labels['night_flux'] = 'night_flux (GB)'

if 'cm_total_mou' in df.columns:
    key_features_data['cm_total_mou'] = df['cm_total_mou'].dropna()
    feature_labels['cm_total_mou'] = 'cm_total_mou (分钟)'

if len(key_features_data) > 0:
    fig, axes = plt.subplots(1, len(key_features_data), figsize=(15, 5))
    if len(key_features_data) == 1:
        axes = [axes]
    
    for idx, (col, data) in enumerate(key_features_data.items()):
        axes[idx].boxplot(data, vert=True)
        axes[idx].set_ylabel(feature_labels[col], fontsize=10)
        axes[idx].set_title(f'{col} 异常值检测', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        # 添加统计信息
        q1, median, q3 = data.quantile([0.25, 0.5, 0.75])
        axes[idx].text(0.02, 0.98, f'中位数: {median:.2f}\nQ1: {q1:.2f}\nQ3: {q3:.2f}', 
                      transform=axes[idx].transAxes, fontsize=8, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(eda_dir / "outliers_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("    [OK] outliers_boxplot.png")

print(f"\n  [OK] EDA可视化报告已生成: {eda_dir}")
print(f"    - missing_values_distribution.png: 缺失值分布")
print(f"    - age_distribution.png: 年龄分布")
print(f"    - arpu_distribution.png: ARPU分布")
print(f"    - app_usage_distribution.png: APP使用分布")
print(f"    - outliers_boxplot.png: 异常值检测")

print("\n" + "=" * 80)
print("[OK] 数据预处理完成!")
print("=" * 80)
print(f"\n交付物:")
print(f"  - cleaned_data.csv: 清洗后数据({len(df):,}条)")
print(f"  - cleaning_report.json: 清洗报告")
print(f"  - EDA可视化报告(5张图表)")
print(f"\n下一步: 运行 02_feature_engineering.py 进行特征工程")

