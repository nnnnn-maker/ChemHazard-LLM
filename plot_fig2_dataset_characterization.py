import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "processed_coarse/full_dataset_coarse_labels.csv"
OUTPUT_DIR = "fig2_outputs"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# =========================
# Global style
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "svg.fonttype": "none",
})


# =========================
# Label definitions
# =========================
label_cols = [
    "label_flammable",
    "label_oxidizing",
    "label_gas_under_pressure",
    "label_corrosive",
    "label_acute_toxicity",
    "label_irritant_harmful",
    "label_cmr",
    "label_stot",
    "label_environmental_hazard",
]


label_display = {
    "label_flammable": "Flammable",
    "label_oxidizing": "Oxidizing",
    "label_gas_under_pressure": "Gas under pressure",
    "label_corrosive": "Corrosive",
    "label_acute_toxicity": "Acute toxicity",
    "label_irritant_harmful": "Irritant / harmful",
    "label_cmr": "CMR",
    "label_stot": "STOT",
    "label_environmental_hazard": "Environmental hazard",
}


abbr = {
    "label_flammable": "Flam",
    "label_oxidizing": "Oxi",
    "label_gas_under_pressure": "Gas",
    "label_corrosive": "Corr",
    "label_acute_toxicity": "Acute",
    "label_irritant_harmful": "Irr/H",
    "label_cmr": "CMR",
    "label_stot": "STOT",
    "label_environmental_hazard": "Env",
}


# =========================
# Load data
# =========================
df = pd.read_csv(INPUT_CSV, low_memory=False)


# Check columns
missing_labels = [x for x in label_cols if x not in df.columns]

if missing_labels:
    raise ValueError(
        f"Missing label columns: {missing_labels}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Only compounds with at least one positive label are used for label-distribution characterization
task_df = df[df[label_cols].sum(axis=1) > 0].copy()
N = len(task_df)

if N == 0:
    raise ValueError("No compounds with positive coarse-grained hazard labels were found.")


# =========================
# (1) Positive label distribution
# =========================
label_counts = task_df[label_cols].sum().sort_values(ascending=False)
label_percent = label_counts / N * 100

label_dist_df = pd.DataFrame({
    "label": [label_display[c] for c in label_counts.index],
    "count": label_counts.values.astype(int),
    "percent": label_percent.values,
})
label_dist_df.to_csv(os.path.join(OUTPUT_DIR, "label_distribution.csv"), index=False)


# =========================
# (2) Number of labels per compound
# =========================
task_df["labels_per_compound"] = task_df[label_cols].sum(axis=1)
cardinality_counts = task_df["labels_per_compound"].value_counts().sort_index()

cardinality_df = pd.DataFrame({
    "num_positive_labels": cardinality_counts.index.astype(int),
    "compound_count": cardinality_counts.values.astype(int),
})
cardinality_df.to_csv(os.path.join(OUTPUT_DIR, "labels_per_compound.csv"), index=False)


# =========================
# Output A: two bar charts in one composite figure
# No a/b/c labels
# =========================
fig = plt.figure(figsize=(11.0, 4.8))
gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.12, 1.0],
    left=0.08,
    right=0.985,
    bottom=0.16,
    top=0.97,
    wspace=0.25,
)

# Left bar chart: positive label distribution
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.barh([label_display[c] for c in label_counts.index], label_counts.values)
ax1.invert_yaxis()
ax1.set_xlabel("Number of positive compounds")
xmax = max(label_counts.values) * 1.23
ax1.set_xlim(0, xmax)

for bar, count, pct in zip(bars1, label_counts.values, label_percent.values):
    ax1.text(
        bar.get_width() + max(label_counts.values) * 0.012,
        bar.get_y() + bar.get_height() / 2,
        f"{int(count)} ({pct:.2f}%)",
        va="center",
        ha="left",
        fontsize=8,
        clip_on=False,
    )

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Right bar chart: number of positive labels per compound
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(cardinality_counts.index.astype(str), cardinality_counts.values)
ax2.set_xlabel("Number of positive labels per compound")
ax2.set_ylabel("Number of compounds")
ax2.set_ylim(0, max(cardinality_counts.values) * 1.10)

for bar in bars2:
    h = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        h + max(cardinality_counts.values) * 0.012,
        f"{int(h)}",
        ha="center",
        va="bottom",
        fontsize=8,
        clip_on=False,
    )

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.savefig(
    os.path.join(OUTPUT_DIR, "Fig2ab_bar_charts.png"),
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
)
fig.savefig(
    os.path.join(OUTPUT_DIR, "Fig2ab_bar_charts.svg"),
    bbox_inches="tight",
    pad_inches=0.02,
)
fig.savefig(
    os.path.join(OUTPUT_DIR, "Fig2ab_bar_charts.pdf"),
    bbox_inches="tight",
    pad_inches=0.02,
)
plt.close(fig)


# =========================
# (3) Jaccard co-occurrence heatmap
# Keep the original heatmap logic as a standalone figure
# =========================
jaccard = np.zeros((len(label_cols), len(label_cols)))

for i, col_i in enumerate(label_cols):
    set_i = df[col_i] == 1
    for j, col_j in enumerate(label_cols):
        set_j = df[col_j] == 1
        intersection = np.logical_and(set_i, set_j).sum()
        union = np.logical_or(set_i, set_j).sum()
        jaccard[i, j] = intersection / union if union > 0 else 0.0

jaccard_df = pd.DataFrame(
    jaccard,
    index=[abbr[c] for c in label_cols],
    columns=[abbr[c] for c in label_cols],
)
jaccard_df.to_csv(os.path.join(OUTPUT_DIR, "label_jaccard_matrix.csv"))

fig = plt.figure(figsize=(8.5, 7.4))
ax = fig.add_subplot(111)
im = ax.imshow(jaccard, cmap="YlGnBu", vmin=0.0, vmax=1.0, interpolation="nearest")

ax.set_xticks(np.arange(len(label_cols)))
ax.set_yticks(np.arange(len(label_cols)))
ax.set_xticklabels([abbr[c] for c in label_cols], rotation=45, ha="right")
ax.set_yticklabels([abbr[c] for c in label_cols])

for i in range(len(label_cols)):
    for j in range(len(label_cols)):
        value = jaccard[i, j]
        text_color = "white" if value >= 0.45 else "black"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Jaccard coefficient")
fig.tight_layout()
fig.savefig(
    os.path.join(OUTPUT_DIR, "Fig2c_label_cooccurrence.png"),
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
)
fig.savefig(
    os.path.join(OUTPUT_DIR, "Fig2c_label_cooccurrence.svg"),
    bbox_inches="tight",
    pad_inches=0.02,
)
fig.savefig(
    os.path.join(OUTPUT_DIR, "Fig2c_label_cooccurrence.pdf"),
    bbox_inches="tight",
    pad_inches=0.02,
)
plt.close(fig)


print("Saved outputs:")
print("  - Fig2ab_bar_charts.(png/svg/pdf)")
print("  - Fig2c_label_cooccurrence.(png/svg/pdf)")
print("No panel labels (a/b/c) are embedded in the graphics.")
