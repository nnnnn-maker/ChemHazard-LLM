#!/usr/bin/env python3
"""Plot Fig. 6: component ablation of ChemHazard-LLM.

The x-axis uses the short configuration IDs A0--A5. Their definitions are
exported with the source data and repeated in a caption text file, keeping the
figure itself compact. The bars contain no numeric value annotations; exact
values remain available in the CSV file for reporting and reproducibility.

Outputs
-------
Fig6_component_ablation.png
Fig6_component_ablation.svg
Fig6_component_ablation.pdf
Fig6_configuration_source_data.csv
Fig6_caption.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MICRO_COLOR = "#4C78A8"
MACRO_COLOR = "#F28E2B"
GRID_COLOR = "#E5E7E9"


CONFIGURATIONS = pd.DataFrame(
    {
        "ID": ["A0", "A1", "A2", "A3", "A4", "A5"],
        "Setting": [
            "Mistral-Instruct zero-shot",
            "LoRA + Rare 2x (no pictogram)",
            "LoRA + pictogram (no Rare 2x)",
            "Single model",
            "Majority-vote ensemble",
            "Full model",
        ],
        "Micro-F1": [0.4175, 0.7511, 0.7485, 0.7571, 0.7655, 0.7653],
        "Macro-F1": [0.2763, 0.7538, 0.7296, 0.7680, 0.7666, 0.7852],
    }
)


CAPTION_EN = (
    "Figure 6. Component ablation results for ChemHazard-LLM. "
    "A0, Mistral-7B-Instruct-v0.1 without GHS-task adaptation; A1, LoRA with rare-label oversampling but without "
    "pictograms; A2, LoRA with pictograms but without rare-label oversampling; "
    "A3, single model with pictograms and rare-label oversampling; A4, "
    "three-model majority-vote ensemble; and A5, the full model with constrained "
    "label-wise vote thresholding. Bars show Micro-F1 and Macro-F1 on the "
    "PubChem test set."
)


def style_axis(ax: plt.Axes) -> None:
    """Apply a restrained manuscript-ready style."""
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_configuration_bars(ax: plt.Axes) -> None:
    """Plot Micro-F1 and Macro-F1 as grouped bars for A0--A5."""
    x = np.arange(len(CONFIGURATIONS))
    micro = CONFIGURATIONS["Micro-F1"].to_numpy()
    macro = CONFIGURATIONS["Macro-F1"].to_numpy()
    width = 0.34

    ax.bar(
        x - width / 2,
        micro,
        width,
        color=MICRO_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Micro-F1",
        zorder=2,
    )
    ax.bar(
        x + width / 2,
        macro,
        width,
        color=MACRO_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Macro-F1",
        zorder=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGURATIONS["ID"].tolist())
    ax.set_ylim(0.0, 0.85)
    ax.set_yticks(np.arange(0.0, 0.81, 0.1))
    ax.set_xlabel("Configuration")
    ax.set_ylabel("F1 score")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    style_axis(ax)


def build_figure(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    configuration_csv = output_dir / "Fig6_configuration_source_data.csv"
    caption_path = output_dir / "Fig6_caption.txt"
    CONFIGURATIONS.to_csv(configuration_csv, index=False)
    caption_path.write_text(f"{CAPTION_EN}\n", encoding="utf-8")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.unicode_minus": True,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    plot_configuration_bars(ax)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.14, top=0.97)

    output_stem = output_dir / "Fig6_component_ablation"
    png_path = output_stem.with_suffix(".png")
    svg_path = output_stem.with_suffix(".svg")
    pdf_path = output_stem.with_suffix(".pdf")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return [configuration_csv, caption_path, png_path, svg_path, pdf_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the ChemHazard-LLM component ablation figure."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fig6_outputs"),
        help="Directory for figure and source-data files (default: fig6_outputs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_figure(args.output_dir)
    print("Generated files:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
