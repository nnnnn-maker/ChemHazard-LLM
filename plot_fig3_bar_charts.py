import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Fig. 3 Zero-shot vs task-adapted performance
# Minimal revision:
#   - Only combine the original two plots into one figure.
#   - Keep the original plotting style and data logic.
#
# Output:
#   Fig3_zero_shot_vs_task_adapted_combined.{png,svg,pdf}
#   Fig3_bar_source_data.csv
# ============================================================


# -----------------------------
# 1. Output directory
# -----------------------------
OUTPUT_DIR = "fig3_bar_outputs"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# -----------------------------
# 2. Source data
# -----------------------------
df = pd.DataFrame({
    "Model": ["Qwen", "Mistral", "Gemma", "ChatGLM", "ChemLLM"],
    "Zero-shot_microPrecision": [0.5211, 0.6251, 0.6456, 0.7492, 0.4650],
    "Zero-shot_microRecall": [0.5418, 0.4244, 0.6085, 0.0874, 0.4012],
    "Zero-shot_microF1": [0.5313, 0.5056, 0.6265, 0.1565, 0.4308],
    "Task-adapted_microPrecision": [0.7703, 0.7731, 0.7444, 0.7443, 0.7854],
    "Task-adapted_microRecall": [0.6967, 0.7038, 0.7023, 0.6877, 0.6408],
    "Task-adapted_microF1": [0.7316, 0.7368, 0.7227, 0.7149, 0.7058],
    "Zero-shot_macroF1": [0.3541, 0.2991, 0.3492, 0.0793, 0.2718],
    "Task-adapted_macroF1": [0.7166, 0.7106, 0.6609, 0.6609, 0.6461],
})

CHEMHAZARD_LLM = {
    "Micro-Precision": 0.7569,
    "Micro-Recall": 0.7739,
    "Micro-F1": 0.7653,
    "Macro-F1": 0.7852,
}

df.to_csv(
    os.path.join(OUTPUT_DIR, "Fig3_bar_source_data.csv"),
    index=False
)


# -----------------------------
# 3. General figure settings
# -----------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "svg.fonttype": "none",   # keep text editable in SVG
})


# -----------------------------
# 4. Plotting helper
# -----------------------------
def plot_f1_dumbbell_on_ax(
    ax,
    data,
    zero_shot_column,
    task_adapted_column,
    final_method_value,
    metric_label,
):
    """Same logic as the original script, but draw on a provided axis."""
    sorted_data = data.sort_values(
        "Task-adapted_macroF1",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    pair_x = np.arange(len(sorted_data))
    zero_shot = sorted_data[zero_shot_column].to_numpy()
    task_adapted = sorted_data[task_adapted_column].to_numpy()

    for x_pos, lower_value, upper_value in zip(pair_x, zero_shot, task_adapted):
        ax.plot(
            [x_pos, x_pos],
            [lower_value, upper_value],
            color="#B8C0C7",
            linewidth=2.0,
            zorder=1,
        )

    ax.scatter(
        pair_x,
        zero_shot,
        s=62,
        color="#5B8DB8",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.scatter(
        pair_x,
        task_adapted,
        s=62,
        color="#E39A45",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    final_x = len(sorted_data)

    ax.scatter(
        [final_x],
        [final_method_value],
        s=90,
        marker="D",
        color="#2F6B5F",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )

    labels = sorted_data["Model"].tolist() + ["ChemHazard-LLM"]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel(metric_label)
    ax.yaxis.grid(True, color="#E5E7E9", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            markerfacecolor="#5B8DB8", markeredgecolor="white", label="Zero-shot",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            markerfacecolor="#E39A45", markeredgecolor="white", label="Task-adapted",
        ),
        Line2D(
            [0], [0], marker="D", linestyle="none", markersize=7,
            markerfacecolor="#2F6B5F", markeredgecolor="white", label="ChemHazard-LLM",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        frameon=True,
        facecolor="white",
        edgecolor="#E5E7E9",
        framealpha=0.95,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.025),
        ncol=1,
        handletextpad=0.4,
        labelspacing=0.6,
        borderpad=0.45,
    )
    legend.get_frame().set_linewidth(0.8)


# -----------------------------
# 5. Combined figure only
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

plot_f1_dumbbell_on_ax(
    ax=axes[0],
    data=df,
    zero_shot_column="Zero-shot_microF1",
    task_adapted_column="Task-adapted_microF1",
    final_method_value=CHEMHAZARD_LLM["Micro-F1"],
    metric_label="Micro-F1",
)

plot_f1_dumbbell_on_ax(
    ax=axes[1],
    data=df,
    zero_shot_column="Zero-shot_macroF1",
    task_adapted_column="Task-adapted_macroF1",
    final_method_value=CHEMHAZARD_LLM["Macro-F1"],
    metric_label="Macro-F1",
)

fig.tight_layout()
fig.savefig(
    os.path.join(OUTPUT_DIR, "Fig3_zero_shot_vs_task_adapted_combined.png"),
    dpi=600,
    bbox_inches="tight",
)
fig.savefig(
    os.path.join(OUTPUT_DIR, "Fig3_zero_shot_vs_task_adapted_combined.svg"),
    bbox_inches="tight",
)
fig.savefig(
    os.path.join(OUTPUT_DIR, "Fig3_zero_shot_vs_task_adapted_combined.pdf"),
    bbox_inches="tight",
)
plt.close(fig)


print("Generated files:")
print(os.path.join(OUTPUT_DIR, "Fig3_bar_source_data.csv"))
print(os.path.join(OUTPUT_DIR, "Fig3_zero_shot_vs_task_adapted_combined.png"))
print(os.path.join(OUTPUT_DIR, "Fig3_zero_shot_vs_task_adapted_combined.svg"))
print(os.path.join(OUTPUT_DIR, "Fig3_zero_shot_vs_task_adapted_combined.pdf"))
