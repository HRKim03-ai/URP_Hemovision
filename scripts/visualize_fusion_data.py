"""
Visualize data distributions for fusion_meta.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 10


def visualize_fusion_data(csv_path: str, output_dir: str = "visualizations") -> None:
    """Visualize distributions of all columns in fusion_meta.csv"""
    df = pd.read_csv(csv_path)
    
    # Fix paths if needed
    old_root = "/home/monetai/Desktop/URP/multimodal_dataset"
    new_root = "/home/monetai/Desktop/URP(학룡)/multimodal_dataset"
    
    if "nail_image_path" in df.columns:
        df["nail_image_path"] = df["nail_image_path"].str.replace(old_root, new_root, regex=False)
    if "conj_image_path" in df.columns:
        df["conj_image_path"] = df["conj_image_path"].str.replace(old_root, new_root, regex=False)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique patients: {df['patient_id'].nunique()}")
    print(f"  Samples per patient: {len(df) / df['patient_id'].nunique():.1f}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Hb value distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df["hb_value"], bins=20, edgecolor="black", alpha=0.7, color="skyblue")
    ax1.set_xlabel("Hb Value (g/dL)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Hb Value Distribution")
    ax1.axvline(df["hb_value"].mean(), color="red", linestyle="--", label=f"Mean: {df['hb_value'].mean():.2f}")
    ax1.axvline(df["hb_value"].median(), color="green", linestyle="--", label=f"Median: {df['hb_value'].median():.2f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Age distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df["age"], bins=20, edgecolor="black", alpha=0.7, color="lightcoral")
    ax2.set_xlabel("Age (years)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Age Distribution")
    ax2.axvline(df["age"].mean(), color="red", linestyle="--", label=f"Mean: {df['age'].mean():.2f}")
    ax2.axvline(df["age"].median(), color="green", linestyle="--", label=f"Median: {df['age'].median():.2f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Side distribution
    ax3 = fig.add_subplot(gs[0, 2])
    side_counts = df["side"].value_counts().sort_index()
    bars = ax3.bar(side_counts.index, side_counts.values, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"])
    ax3.set_xlabel("Side")
    ax3.set_ylabel("Count")
    ax3.set_title("Side Distribution (LL/LR/RL/RR)")
    ax3.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 4. Gender distribution
    ax4 = fig.add_subplot(gs[1, 0])
    gender_counts = df["gender"].value_counts()
    bars = ax4.bar(gender_counts.index, gender_counts.values, color=["#FFB6C1", "#87CEEB"])
    ax4.set_xlabel("Gender")
    ax4.set_ylabel("Count")
    ax4.set_title("Gender Distribution")
    ax4.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 5. Patient ID distribution (samples per patient)
    ax5 = fig.add_subplot(gs[1, 1])
    patient_counts = df["patient_id"].value_counts().sort_index()
    ax5.bar(range(len(patient_counts)), patient_counts.values, color="mediumpurple", alpha=0.7)
    ax5.set_xlabel("Patient ID (index)")
    ax5.set_ylabel("Number of Samples")
    ax5.set_title(f"Samples per Patient (Total: {len(patient_counts)} patients)")
    ax5.grid(True, alpha=0.3, axis="y")
    ax5.set_xticks(range(0, len(patient_counts), max(1, len(patient_counts)//10)))
    ax5.set_xticklabels([patient_counts.index[i] for i in range(0, len(patient_counts), max(1, len(patient_counts)//10))], rotation=45)
    
    # 6. Hb vs Age scatter
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(df["age"], df["hb_value"], alpha=0.6, c=df["hb_value"], cmap="viridis", s=50)
    ax6.set_xlabel("Age (years)")
    ax6.set_ylabel("Hb Value (g/dL)")
    ax6.set_title("Hb Value vs Age")
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label="Hb (g/dL)")
    
    # 7. Hb distribution by Gender
    ax7 = fig.add_subplot(gs[2, 0])
    for gender in df["gender"].unique():
        subset = df[df["gender"] == gender]
        ax7.hist(subset["hb_value"], bins=15, alpha=0.6, label=gender, edgecolor="black")
    ax7.set_xlabel("Hb Value (g/dL)")
    ax7.set_ylabel("Frequency")
    ax7.set_title("Hb Distribution by Gender")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Hb distribution by Side
    ax8 = fig.add_subplot(gs[2, 1])
    for side in sorted(df["side"].unique()):
        subset = df[df["side"] == side]
        ax8.hist(subset["hb_value"], bins=15, alpha=0.5, label=side, edgecolor="black")
    ax8.set_xlabel("Hb Value (g/dL)")
    ax8.set_ylabel("Frequency")
    ax8.set_title("Hb Distribution by Side")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Statistics table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    
    stats_text = f"""
Dataset Statistics

Total Samples: {len(df)}
Unique Patients: {df['patient_id'].nunique()}

Hb Value:
  Mean: {df['hb_value'].mean():.2f} g/dL
  Std:  {df['hb_value'].std():.2f} g/dL
  Min:  {df['hb_value'].min():.2f} g/dL
  Max:  {df['hb_value'].max():.2f} g/dL
  Median: {df['hb_value'].median():.2f} g/dL

Age:
  Mean: {df['age'].mean():.2f} years
  Std:  {df['age'].std():.2f} years
  Min:  {df['age'].min():.2f} years
  Max:  {df['age'].max():.2f} years
  Median: {df['age'].median():.2f} years

Gender:
  M: {len(df[df['gender'] == 'M'])} ({100*len(df[df['gender'] == 'M'])/len(df):.1f}%)
  F: {len(df[df['gender'] == 'F'])} ({100*len(df[df['gender'] == 'F'])/len(df):.1f}%)

Side:
  LL: {len(df[df['side'] == 'LL'])}
  LR: {len(df[df['side'] == 'LR'])}
  RL: {len(df[df['side'] == 'RL'])}
  RR: {len(df[df['side'] == 'RR'])}
"""
    ax9.text(0.1, 0.5, stats_text, fontsize=10, family="monospace", verticalalignment="center")
    
    plt.suptitle("Fusion Dataset (fusion_meta.csv) Data Distributions", fontsize=16, fontweight="bold", y=0.995)
    
    output_file = output_path / "fusion_data_distributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_file}")
    
    # Also save individual plots
    print("\nGenerating individual plots...")
    
    # Individual: Hb distribution
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df["hb_value"], bins=20, edgecolor="black", alpha=0.7, color="skyblue")
    ax.set_xlabel("Hb Value (g/dL)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Hb Value Distribution", fontsize=14, fontweight="bold")
    ax.axvline(df["hb_value"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {df['hb_value'].mean():.2f}")
    ax.axvline(df["hb_value"].median(), color="green", linestyle="--", linewidth=2, label=f"Median: {df['hb_value'].median():.2f}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "hb_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Individual: Age distribution
    fig3, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df["age"], bins=20, edgecolor="black", alpha=0.7, color="lightcoral")
    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Age Distribution", fontsize=14, fontweight="bold")
    ax.axvline(df["age"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {df['age'].mean():.2f}")
    ax.axvline(df["age"].median(), color="green", linestyle="--", linewidth=2, label=f"Median: {df['age'].median():.2f}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "age_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Individual: Side distribution
    fig4, ax = plt.subplots(figsize=(8, 6))
    side_counts = df["side"].value_counts().sort_index()
    bars = ax.bar(side_counts.index, side_counts.values, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"])
    ax.set_xlabel("Side", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Side Distribution (LL/LR/RL/RR)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path / "side_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Individual: Gender distribution
    fig5, ax = plt.subplots(figsize=(8, 6))
    gender_counts = df["gender"].value_counts()
    bars = ax.bar(gender_counts.index, gender_counts.values, color=["#FFB6C1", "#87CEEB"])
    ax.set_xlabel("Gender", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Gender Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path / "gender_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Individual: Hb vs Age scatter
    fig6, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["age"], df["hb_value"], alpha=0.6, c=df["hb_value"], cmap="viridis", s=50)
    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel("Hb Value (g/dL)", fontsize=12)
    ax.set_title("Hb Value vs Age", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Hb (g/dL)")
    plt.tight_layout()
    plt.savefig(output_path / "hb_vs_age.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"All visualizations saved to: {output_path}/")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize fusion_meta.csv data distributions")
    parser.add_argument(
        "--csv",
        type=str,
        default="multimodal_dataset/fusion_meta.csv",
        help="Path to fusion_meta.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations",
        help="Output directory for visualizations",
    )
    args = parser.parse_args()
    
    visualize_fusion_data(args.csv, args.output)


if __name__ == "__main__":
    main()

