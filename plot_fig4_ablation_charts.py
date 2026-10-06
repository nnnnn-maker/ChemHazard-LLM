import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Fig. 4 Controlled ablation charts
# Minimal revision:
#   - Only combine the original two plots into one figure.
#   - Keep the original data, colors, axes, legends, and styling.
#
# Output:
#   Fig4_controlled_ablation_combined.{png,svg,pdf}
#   Fig4_incremental_fields_source_data.csv
#   Fig4_pictogram_control_source_data.csv
# ============================================================


# -----------------------------
# 1. Output directory
# -----------------------------
OUTPUT_DIR = "fig4_outputs"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# -----------------------------
# 2. Source data
#    Current values from Table 4 in the manuscript.
#    Replace these values if your table changes later.
# -----------------------------
incremental_df = pd.DataFrame({
    "Setting": [
        "structure_only",
        "structure_physchem",
        "structure_physchem_toxicity",
    ],
    "Display": [
        "Structure only",
        "Structure +\nphysicochemical",
        "Structure + physicochemical\n+ toxicity",
    ],
    "Micro-F1": [0.6751, 0.7247, 0.7236],
    "Macro-F1": [0.6064, 0.7045, 0.7167],
})

pictogram_df = pd.DataFrame({
    "Setting": [
        "full_nonleaking",
        "full_nonleaking_pictogram",
        "full_nonleaking_shuffled_pictogram",
        "pictogram_only",
    ],
    "Display": [
        "Full non-leaking",
        "Full non-leaking\n+ correct pictogram",
        "Full non-leaking\n+ shuffled pictogram",
        "Pictogram only",
    ],
    "Micro-F1": [0.7180, 0.7331, 0.7160, 0.6301],
    "Macro-F1": [0.7021, 0.7174, 0.6988, 0.3526],
})

incremental_df.to_csv(
    os.path.join(OUTPUT_DIR, "Fig4_incremental_fields_source_data.csv"),
    index=False
)

pictogram_df.to_csv(
    os.path.join(OUTPUT_DIR, "Fig4_pictogram_control_source_data.csv"),
    index=False
)


# -----------------------------
# 3. General plotting settings
# -----------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "svg.fonttype": "none",
})


# -----------------------------
# 4. Combined figure only
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

# Left: original Fig. 4(a)
x = np.arange(len(incremental_df))
ax = axes[0]

ax.plot(
    x,
    incremental_df["Micro-F1"].tolist(),
    marker="o",
    label="Micro-F1",
    color="#5B8DB8",
    linewidth=2.0,
)

ax.plot(
    x,
    incremental_df["Macro-F1"].tolist(),
    marker="o",
    label="Macro-F1",
    color="#E39A45",
    linewidth=2.0,
)

ax.set_xticks(x)
ax.set_xticklabels(incremental_df["Display"].tolist(), rotation=0, ha="center")
ax.set_ylim(0.55, 0.78)
ax.legend(frameon=False)
ax.yaxis.grid(True, color="#E5E7E9", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# Right: original Fig. 4(b)
x = np.arange(len(pictogram_df))
width = 0.24
ax = axes[1]

ax.bar(
    x - width / 2,
    pictogram_df["Micro-F1"].tolist(),
    width,
    label="Micro-F1",
    color="#5B8DB8",
)

ax.bar(
    x + width / 2,
    pictogram_df["Macro-F1"].tolist(),
    width,
    label="Macro-F1",
    color="#E39A45",
)

ax.set_xticks(x)
ax.set_xticklabels(pictogram_df["Display"].tolist())
ax.set_ylim(0.0, 0.80)
ax.legend(frameon=False)
ax.yaxis.grid(True, color="#E5E7E9", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.tight_layout()

fig4_png = os.path.join(OUTPUT_DIR, "Fig4_controlled_ablation_combined.png")
fig4_svg = os.path.join(OUTPUT_DIR, "Fig4_controlled_ablation_combined.svg")
fig4_pdf = os.path.join(OUTPUT_DIR, "Fig4_controlled_ablation_combined.pdf")

fig.savefig(fig4_png, dpi=600, bbox_inches="tight")
fig.savefig(fig4_svg, bbox_inches="tight")
fig.savefig(fig4_pdf, bbox_inches="tight")
plt.close(fig)


# -----------------------------
# 5. Console output
# -----------------------------
print("Generated files:")
print(os.path.join(OUTPUT_DIR, "Fig4_incremental_fields_source_data.csv"))
print(os.path.join(OUTPUT_DIR, "Fig4_pictogram_control_source_data.csv"))
print(fig4_png)
print(fig4_svg)
print(fig4_pdf)
