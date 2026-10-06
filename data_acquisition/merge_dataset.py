"""
合并 PubChem 数据集
将 compounds.csv, ghs_data.csv, extended_properties.csv 合并为 full_dataset.csv
"""

import pandas as pd
import os


def merge_datasets(data_dir="pubchem_data", output_file="full_dataset.csv"):
    """
    合并三个数据文件为完整数据集

    Args:
        data_dir: 数据目录
        output_file: 输出文件名
    """
    print("=" * 50)
    print("开始合并数据集")
    print("=" * 50)

    # 1. 加载基础化合物数据
    compounds_path = f"{data_dir}/compounds.csv"
    print(f"\n加载基础数据: {compounds_path}")
    df_compounds = pd.read_csv(compounds_path)
    print(f"  记录数: {len(df_compounds)}")
    print(f"  字段: {list(df_compounds.columns)}")

    # 2. 加载 GHS 危险性数据
    ghs_path = f"{data_dir}/ghs_data.csv"
    print(f"\n加载 GHS 数据: {ghs_path}")
    df_ghs = pd.read_csv(ghs_path)
    print(f"  记录数: {len(df_ghs)}")
    print(f"  有 GHS 数据的记录: {df_ghs['Has_GHS_Data'].sum()}")

    # 3. 加载扩展属性数据
    extended_path = f"{data_dir}/extended_properties.csv"
    print(f"\n加载扩展属性: {extended_path}")
    df_extended = pd.read_csv(extended_path)
    print(f"  记录数: {len(df_extended)}")

    # 4. 合并数据
    print("\n合并数据...")

    # 先合并 compounds 和 ghs
    df_merged = pd.merge(df_compounds, df_ghs, on='CID', how='left')
    print(f"  合并 compounds + ghs: {len(df_merged)} 条")

    # 再合并 extended_properties
    df_merged = pd.merge(df_merged, df_extended, on='CID', how='left')
    print(f"  合并 + extended: {len(df_merged)} 条")

    # 5. 数据统计
    print("\n" + "=" * 50)
    print("数据统计")
    print("=" * 50)
    print(f"总记录数: {len(df_merged)}")
    print(f"总字段数: {len(df_merged.columns)}")
    print(f"\n字段列表:")
    for i, col in enumerate(df_merged.columns, 1):
        print(f"  {i:2d}. {col}")

    # 统计有效数据
    if 'Has_GHS_Data' in df_merged.columns:
        ghs_count = df_merged['Has_GHS_Data'].sum()
        print(f"\n有 GHS 数据: {ghs_count} ({ghs_count/len(df_merged)*100:.1f}%)")

    if 'Has_Physical_Data' in df_merged.columns:
        phys_count = df_merged['Has_Physical_Data'].sum()
        print(f"有物理性质数据: {phys_count} ({phys_count/len(df_merged)*100:.1f}%)")

    if 'Has_Toxicity_Data' in df_merged.columns:
        tox_count = df_merged['Has_Toxicity_Data'].sum()
        print(f"有毒性数据: {tox_count} ({tox_count/len(df_merged)*100:.1f}%)")

    # 6. 保存合并后的数据
    output_path = f"{data_dir}/{output_file}"
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n已保存到: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return df_merged


def main():
    import argparse

    parser = argparse.ArgumentParser(description='合并 PubChem 数据集')
    parser.add_argument('--data_dir', type=str, default='pubchem_data',
                       help='数据目录 (默认: pubchem_data)')
    parser.add_argument('--output', type=str, default='full_dataset.csv',
                       help='输出文件名 (默认: full_dataset.csv)')

    args = parser.parse_args()

    merge_datasets(args.data_dir, args.output)

    print("\n合并完成！")


if __name__ == "__main__":
    main()
