"""
Z世代客群分析 - 一键执行脚本
作者: 算法工程师
日期: 2024-12
功能: 按顺序执行所有算法步骤
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_script(script_path, step_name):
    """执行Python脚本"""
    print("\n" + "=" * 80)
    print(f"开始执行: {step_name}")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print(f"✓ {step_name} 完成! 耗时: {duration:.1f}秒")
        print("=" * 80)
        
        return True
    
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"✗ {step_name} 失败!")
        print("=" * 80)
        print(f"错误: {e}")
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("Z世代客群分析 - 算法模块完整执行")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n执行计划:")
    print("  Step 1: 数据预处理 (01_data_preprocessing.py)")
    print("  Step 2: 特征工程 (02_feature_engineering.py)")
    print("  Step 3: 规则分群 (03_rule_based_segmentation.py)")
    print("  Step 4: XGBoost分类 (04_xgboost_classification_focal.py)")
    print("=" * 80)
    
    scripts_dir = Path(__file__).parent / "algorithm" / "scripts"
    
    # 定义执行步骤
    steps = [
        (scripts_dir / "01_data_preprocessing.py", "Step 1: 数据预处理"),
        (scripts_dir / "02_feature_engineering.py", "Step 2: 特征工程"),
        (scripts_dir / "03_rule_based_segmentation.py", "Step 3: 规则分群"),
        (scripts_dir / "04_xgboost_classification_focal.py", "Step 4: XGBoost分类")
    ]
    
    # 记录开始时间
    total_start_time = datetime.now()
    
    # 逐步执行
    for script_path, step_name in steps:
        if not script_path.exists():
            print(f"\n✗ 错误: 脚本文件不存在: {script_path}")
            sys.exit(1)
        
        success = run_script(script_path, step_name)
        
        if not success:
            print(f"\n✗ 执行中止于: {step_name}")
            sys.exit(1)
    
    # 计算总耗时
    total_end_time = datetime.now()
    total_duration = (total_end_time - total_start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("✓✓✓ 全部流程执行完成! ✓✓✓")
    print("=" * 80)
    print(f"总耗时: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
    print(f"完成时间: {total_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n交付物汇总:")
    print("  📁 模型文件 (3个):")
    print("     - algorithm/models/label_encoder.pkl")
    print("     - algorithm/models/xgboost_model.pkl")
    print("     - algorithm/models/feature_scaler.pkl")
    
    print("\n  📁 数据文件 (7个):")
    print("     - algorithm/outputs/data/cleaned_data.csv")
    print("     - algorithm/outputs/data/features_data.csv")
    print("     - algorithm/outputs/data/segments_data.json")
    print("     - algorithm/outputs/data/segment_profiles.json")
    print("     - algorithm/outputs/data/model_metrics.json")
    print("     - algorithm/outputs/data/feature_importance.json")
    print("     - algorithm/outputs/data/training_log.json")
    
    print("\n  📁 可视化图表 (15张):")
    print("     分群相关 (3张):")
    print("       - segment_distribution_pie.png")
    print("       - app_preference_heatmap.png")
    print("       - ratio_binning_mapping.png")
    print("     分类相关 (5张):")
    print("       - confusion_matrix.png")
    print("       - feature_importance.png")
    print("       - learning_curve.png")
    print("       - roc_curves.png")
    print("       - precision_recall.png")
    print("     客群画像 (7张):")
    print("       - segment_1_profile.png ~ segment_6_profile.png")
    print("       - segment_comparison.png")
    
    print("\n" + "=" * 80)
    print("🎉 恭喜! 算法模块交付完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()










