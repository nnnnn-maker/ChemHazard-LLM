import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from audit_panel_alignment import require_matplotlib_panel_alignment


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
    }
)


WIDTH_MM = 183
HEIGHT_MM = 92

TEXT = "#20262E"
MUTED = "#59636E"
ARROW = "#4B5560"
WHITE = "#FFFFFF"

STAGES = [
    {"edge": "#3F78B5", "fill": "#F3F7FC"},
    {"edge": "#4F8F5B", "fill": "#F4F8F3"},
    {"edge": "#C98B18", "fill": "#FCF8EE"},
    {"edge": "#7860AA", "fill": "#F7F4FB"},
    {"edge": "#29889D", "fill": "#F1F8FA"},
]
RED = "#C84D4D"
RED_FILL = "#FFF5F4"


def rounded_box(ax, x, y, width, height, edge, fill=WHITE, radius=0.045, lw=0.8):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def panel_frame(ax, panel_id, title, style, title_fontsize=8.5):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    rounded_box(ax, 0.01, 0.01, 0.98, 0.98, style["edge"], style["fill"], radius=0.07, lw=0.9)
    ax.text(
        0.055,
        0.955,
        panel_id,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.18,
        0.955,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=title_fontsize,
        fontweight="bold",
        linespacing=1.30,
        color=TEXT,
    )


def card(ax, x, y, width, height, edge, title, body=None, fill=WHITE, title_color=None):
    rounded_box(ax, x, y, width, height, edge, fill, radius=0.045, lw=0.8)
    ax.text(
        x + width / 2,
        y + height - 0.045,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        linespacing=1.08,
        color=title_color or edge,
    )
    if body:
        title_lines = title.count("\n") + 1
        ax.text(
            x + 0.045,
            y + height - 0.055 - 0.052 * title_lines,
            body,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            linespacing=1.12,
            color=TEXT,
        )


def vertical_arrow(ax, x, y_top, y_bottom, color=ARROW):
    ax.add_patch(
        FancyArrowPatch(
            (x, y_top),
            (x, y_bottom),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.8,
            color=color,
            clip_on=False,
        )
    )


def label_matrix(ax, x, y, size, color):
    step = size / 3
    xs = [x + (col + 0.5) * step for row in range(3) for col in range(3)]
    ys = [y + (2.5 - row) * step for row in range(3) for col in range(3)]
    ax.scatter(xs, ys, s=15, marker="s", color=color, alpha=0.68, transform=ax.transAxes, clip_on=False)


def build_figure():
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor=WHITE)
    grid = fig.add_gridspec(
        1,
        5,
        left=0.012,
        right=0.988,
        bottom=0.025,
        top=0.975,
        wspace=0.11,
    )
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(5)]

    titles = [
        "Data and label\nconstruction",
        "Evidence control\nand leakage audit",
        "ChemHazard-LLM\nadaptation",
        "Ensemble and\nlong-tail decision",
        "Prediction and\nevaluation",
    ]
    for ax, panel_id, title, style in zip(axes, "abcde", titles, STAGES):
        panel_frame(ax, panel_id, title, style, title_fontsize=8.5)

    # Panel a: data and label construction.
    ax = axes[0]
    color = STAGES[0]["edge"]
    card(
        ax,
        0.08,
        0.64,
        0.84,
        0.22,
        color,
        "PubChem records",
        "Structure\nPhysicochemical\nToxicity / safety data",
    )
    vertical_arrow(ax, 0.50, 0.63, 0.56)
    card(ax, 0.13, 0.44, 0.74, 0.11, color, "Coarse GHS\nmapping")
    vertical_arrow(ax, 0.50, 0.44, 0.36)
    card(ax, 0.10, 0.17, 0.80, 0.18, color, "Nine-label\nmulti-label dataset")

    # Panel b: evidence control and overlap auditing.
    ax = axes[1]
    color = STAGES[1]["edge"]
    card(
        ax,
        0.07,
        0.62,
        0.86,
        0.24,
        color,
        "Retained inputs",
        "Structure\nPhysicochemical\nToxicity fields\nGHS pictograms",
    )
    vertical_arrow(ax, 0.50, 0.61, 0.55)
    card(
        ax,
        0.07,
        0.35,
        0.86,
        0.19,
        RED,
        "Excluded evidence",
        "H-statements /\nH-codes\nDirect GHS text",
        fill=RED_FILL,
        title_color=RED,
    )
    vertical_arrow(ax, 0.50, 0.34, 0.28)
    card(
        ax,
        0.10,
        0.11,
        0.80,
        0.17,
        color,
        "Overlap audit",
        "CID / SMILES /\nscaffold overlap",
    )

    # Panel c: instruction tuning and structured output.
    ax = axes[2]
    color = STAGES[2]["edge"]
    card(
        ax,
        0.09,
        0.67,
        0.82,
        0.19,
        color,
        "Chat-style SFT",
        "Structured prompt\nTarget JSON",
    )
    vertical_arrow(ax, 0.50, 0.66, 0.58)
    card(
        ax,
        0.09,
        0.39,
        0.82,
        0.19,
        color,
        "Mistral-7B-Instruct-v0.1\n+ LoRA",
        "Task adaptation",
    )
    vertical_arrow(ax, 0.50, 0.38, 0.31)
    card(ax, 0.13, 0.14, 0.74, 0.16, color, "Fixed nine-key\nJSON output")

    # Panel d: long-tail handling and ensemble decision.
    ax = axes[3]
    color = STAGES[3]["edge"]
    rare_card_y = 0.74
    rare_card_height = 0.12
    card(ax, 0.10, rare_card_y, 0.80, rare_card_height, color, "Rare-label\noversampling")
    seed_label_y = 0.695
    seed_box_y = 0.58
    ax.text(0.50, seed_label_y, "Training seeds", transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color=MUTED)
    seed_x = [0.08, 0.38, 0.68]
    for x, seed in zip(seed_x, ["123", "777", "2025"]):
        rounded_box(ax, x, seed_box_y, 0.24, 0.10, color, WHITE, radius=0.035, lw=0.8)
        ax.text(
            x + 0.12,
            seed_box_y + 0.05,
            seed,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=TEXT,
        )
    for x in [0.20, 0.50, 0.80]:
        ax.add_patch(
            FancyArrowPatch(
                (x, seed_box_y - 0.005),
                (0.50, 0.47),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=6,
                linewidth=0.7,
                color=ARROW,
            )
        )
    card(ax, 0.14, 0.36, 0.72, 0.10, color, "Label-wise voting")
    vertical_arrow(ax, 0.50, 0.35, 0.29)
    card(
        ax,
        0.08,
        0.08,
        0.84,
        0.23,
        color,
        "Constrained\nvote thresholds",
        "Selected: 1/3\nOthers: 2/3",
    )

    # Panel e: final output and validation routes.
    ax = axes[4]
    color = STAGES[4]["edge"]
    card(ax, 0.09, 0.66, 0.82, 0.20, color, "Final nine-label\nprediction")
    label_matrix(ax, 0.47, 0.675, 0.06, color)
    vertical_arrow(ax, 0.50, 0.65, 0.57)
    card(
        ax,
        0.08,
        0.25,
        0.84,
        0.31,
        color,
        "Evaluation",
        "Baselines\nControlled ablation\nScaffold split\nBootstrap 95% CI",
    )

    fig.canvas.draw()
    for left_ax, right_ax in zip(axes[:-1], axes[1:]):
        left = left_ax.get_position()
        right = right_ax.get_position()
        fig.add_artist(
            FancyArrowPatch(
                (left.x1 + 0.002, 0.50),
                (right.x0 - 0.002, 0.50),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.9,
                color=ARROW,
                zorder=20,
            )
        )

    return fig, axes


def main():
    parser = argparse.ArgumentParser(description="Render the ChemHazard-LLM framework figure.")
    parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    out_base = out_dir / "figure_1_chemhazard_framework"
    fig, axes = build_figure()

    require_matplotlib_panel_alignment(
        fig,
        axes=axes,
        panel_ids=list("abcde"),
        row_groups=[list("abcde")],
        json_out=str(out_base.with_suffix(".alignment-layout.json")),
        overlay_svg=str(out_base.with_suffix(".alignment-overlay.svg")),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )

    fig.savefig(out_base.with_suffix(".svg"), facecolor=WHITE)
    fig.savefig(out_base.with_suffix(".pdf"), facecolor=WHITE)
    fig.savefig(out_base.with_suffix(".png"), dpi=600, facecolor=WHITE)
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, facecolor=WHITE)
    plt.close(fig)


if __name__ == "__main__":
    main()
