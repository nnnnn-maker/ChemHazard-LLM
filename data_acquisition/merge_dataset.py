"""
Merge PubChem datasets.
Combine compounds.csv, ghs_data.csv, and extended_properties.csv into full_dataset.csv.
"""

import pandas as pd
import os


def merge_datasets(data_dir="pubchem_data", output_file="full_dataset.csv"):
    """
    Merge the three source files into a complete dataset.

    Args:
        data_dir: Directory containing the source files.
        output_file: Output file name.
    """
    print("=" * 50)
    print("Starting dataset merge")
    print("=" * 50)

    # 1. Load basic compound data.
    compounds_path = f"{data_dir}/compounds.csv"
    print(f"\nLoading basic data: {compounds_path}")
    df_compounds = pd.read_csv(compounds_path)
    print(f"  Records: {len(df_compounds)}")
    print(f"  Columns: {list(df_compounds.columns)}")

    # 2. Load GHS hazard data.
    ghs_path = f"{data_dir}/ghs_data.csv"
    print(f"\nLoading GHS data: {ghs_path}")
    df_ghs = pd.read_csv(ghs_path)
    print(f"  Records: {len(df_ghs)}")
    print(f"  Records with GHS data: {df_ghs['Has_GHS_Data'].sum()}")

    # 3. Load extended property data.
    extended_path = f"{data_dir}/extended_properties.csv"
    print(f"\nLoading extended properties: {extended_path}")
    df_extended = pd.read_csv(extended_path)
    print(f"  Records: {len(df_extended)}")

    # 4. Merge the datasets.
    print("\nMerging data...")

    # Merge compounds and GHS data first.
    df_merged = pd.merge(df_compounds, df_ghs, on='CID', how='left')
    print(f"  Compounds + GHS merged: {len(df_merged)} records")

    # Then merge extended properties.
    df_merged = pd.merge(df_merged, df_extended, on='CID', how='left')
    print(f"  Extended properties merged: {len(df_merged)} records")

    # 5. Summarize the merged data.
    print("\n" + "=" * 50)
    print("Dataset statistics")
    print("=" * 50)
    print(f"Total records: {len(df_merged)}")
    print(f"Total columns: {len(df_merged.columns)}")
    print("\nColumns:")
    for i, col in enumerate(df_merged.columns, 1):
        print(f"  {i:2d}. {col}")

    # Count records with available annotations or properties.
    if 'Has_GHS_Data' in df_merged.columns:
        ghs_count = df_merged['Has_GHS_Data'].sum()
        print(f"\nRecords with GHS data: {ghs_count} ({ghs_count/len(df_merged)*100:.1f}%)")

    if 'Has_Physical_Data' in df_merged.columns:
        phys_count = df_merged['Has_Physical_Data'].sum()
        print(f"Records with physical data: {phys_count} ({phys_count/len(df_merged)*100:.1f}%)")

    if 'Has_Toxicity_Data' in df_merged.columns:
        tox_count = df_merged['Has_Toxicity_Data'].sum()
        print(f"Records with toxicity data: {tox_count} ({tox_count/len(df_merged)*100:.1f}%)")

    # 6. Save the merged dataset.
    output_path = f"{data_dir}/{output_file}"
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return df_merged


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Merge PubChem datasets')
    parser.add_argument('--data_dir', type=str, default='pubchem_data',
                       help='Data directory (default: pubchem_data)')
    parser.add_argument('--output', type=str, default='full_dataset.csv',
                       help='Output file name (default: full_dataset.csv)')

    args = parser.parse_args()

    merge_datasets(args.data_dir, args.output)

    print("\nMerge complete!")


if __name__ == "__main__":
    main()
