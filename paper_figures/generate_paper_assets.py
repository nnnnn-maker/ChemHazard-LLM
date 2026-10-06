from __future__ import annotations

import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))


MAIN_RESULTS = [
    {
        "experiment": "zero-shot Mistral",
        "setting": "Mistral-Instruct zero-shot, no GHS-task LoRA",
        "precision": 0.6094,
        "recall": 0.3176,
        "micro_f1": 0.4175,
        "macro_f1": 0.2763,
        "hamming_loss": 0.2271,
        "subset_accuracy": 0.0727,
    },
    {
        "experiment": "focus_cmr_stot",
        "setting": "No direct GHS weak evidence",
        "precision": 0.7461,
        "recall": 0.7624,
        "micro_f1": 0.7542,
        "macro_f1": 0.7438,
        "hamming_loss": 0.1249,
        "subset_accuracy": 0.3391,
    },
    {
        "experiment": "no-GHS + rare2x",
        "setting": "No direct GHS evidence + rare-label oversampling",
        "precision": 0.7601,
        "recall": 0.7424,
        "micro_f1": 0.7511,
        "macro_f1": 0.7538,
        "hamming_loss": 0.1261,
        "subset_accuracy": 0.3348,
    },
    {
        "experiment": "pictograms only",
        "setting": "GHS pictograms, no rare2x",
        "precision": 0.7738,
        "recall": 0.7248,
        "micro_f1": 0.7485,
        "macro_f1": 0.7296,
        "hamming_loss": 0.1249,
        "subset_accuracy": 0.3348,
    },
    {
        "experiment": "pictograms + rare2x",
        "setting": "GHS pictograms + rare-label oversampling",
        "precision": 0.7587,
        "recall": 0.7555,
        "micro_f1": 0.7571,
        "macro_f1": 0.7680,
        "hamming_loss": 0.1243,
        "subset_accuracy": 0.3348,
    },
    {
        "experiment": "pictograms + rare2x + ensemble3",
        "setting": "Three-seed majority-vote ensemble",
        "precision": 0.7684,
        "recall": 0.7627,
        "micro_f1": 0.7655,
        "macro_f1": 0.7666,
        "hamming_loss": 0.1198,
        "subset_accuracy": 0.3486,
    },
    {
        "experiment": "constrained calibration",
        "setting": "Ensemble + 1/3 threshold for oxidizing, gas, CMR",
        "precision": 0.7569,
        "recall": 0.7739,
        "micro_f1": 0.7653,
        "macro_f1": 0.7852,
        "hamming_loss": 0.1217,
        "subset_accuracy": 0.3382,
    },
]


CROSS_LLM = [
    {
        "model": "Mistral-7B-Instruct-v0.1",
        "setting": "LoRA, pictograms + rare2x, seed123",
        "precision": 0.7587,
        "recall": 0.7555,
        "micro_f1": 0.7571,
        "macro_f1": 0.7680,
        "hamming_loss": 0.1243,
        "subset_accuracy": 0.3348,
        "parse_failures": 0,
    },
    {
        "model": "Qwen2.5-7B-Instruct",
        "setting": "LoRA, pictograms + rare2x, seed123",
        "precision": 0.7569,
        "recall": 0.7435,
        "micro_f1": 0.7501,
        "macro_f1": 0.7507,
        "hamming_loss": 0.1270,
        "subset_accuracy": 0.3296,
        "parse_failures": 0,
    },
    {
        "model": "Morgan+PhysChem RF",
        "setting": "Conventional baseline, same main split",
        "precision": 0.6842,
        "recall": 0.7799,
        "micro_f1": 0.7289,
        "macro_f1": 0.6593,
        "hamming_loss": 0.1487,
        "subset_accuracy": 0.2621,
        "parse_failures": 0,
    },
]


QWEN_CI = [
    ("subset_accuracy", 0.3296, 0.3028, 0.3581),
    ("micro_f1", 0.7501, 0.7366, 0.7641),
    ("macro_f1", 0.7507, 0.7048, 0.7823),
    ("hamming_loss", 0.1270, 0.1202, 0.1333),
]


BASELINES = [
    ("Morgan+PhysChem RF", 0.2621, 0.7289, 0.6593, 0.6842, 0.7799, 0.1487),
    ("PhysChem RF", 0.2240, 0.7012, 0.6256, 0.6835, 0.7199, 0.1572),
    ("Morgan RF", 0.1349, 0.6621, 0.5313, 0.5740, 0.7822, 0.2046),
    ("PhysChem LogReg", 0.0407, 0.5484, 0.4653, 0.4700, 0.6580, 0.2779),
    ("Morgan LogReg", 0.0978, 0.5458, 0.4593, 0.5096, 0.5876, 0.2507),
]


SCAFFOLD = [
    ("Mistral LoRA single", 0.2394, 0.7011, 0.5647, 0.7134, 0.6892, 0.1495),
    ("Mistral LoRA ensemble3", 0.2300, 0.7000, 0.5563, 0.7180, 0.6828, 0.1489),
    ("Morgan+PhysChem RF", 0.1637, 0.6589, 0.4810, 0.6605, 0.6573, 0.1731),
    ("PhysChem RF", 0.1662, 0.6559, 0.5071, 0.6718, 0.6407, 0.1710),
    ("Morgan RF", 0.0999, 0.6088, 0.3883, 0.5440, 0.6911, 0.2260),
]


RARE_LABELS = [
    ("oxidizing", 0.7692, 0.8750, 0.7143, 1.0000, 0.8333, 0.7778),
    ("gas_under_pressure", 0.8800, 0.9286, 0.8462, 1.0000, 0.9167, 0.8667),
    ("cmr", 0.6240, 0.6374, 0.5657, 0.6970, 0.6957, 0.5872),
]


LEAKAGE = [
    ("Main split", "CID train-test overlap", 0),
    ("Main split", "SMILES train-test overlap", 22),
    ("Main split", "Scaffold train-test overlap", 185),
    ("Scaffold split", "CID train-test overlap", 0),
    ("Scaffold split", "SMILES train-test overlap", 0),
    ("Scaffold split", "Scaffold train-test overlap", 0),
]


ENV_ERROR = [
    ("ambiguous_boundary", 46),
    ("clear_model_error", 33),
    ("missing_evidence", 19),
    ("label_noise", 2),
]


def write_csv(path: Path, rows: list[dict] | list[tuple], header: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            if isinstance(row, dict):
                writer.writerow([row.get(name, "") for name in header])
            else:
                writer.writerow(row)


def write_markdown_tables() -> None:
    lines = [
        "# Final Tables for Manuscript",
        "",
        "Values are rounded to four decimals. `Precision`, `Recall`, and `micro-F1` are micro-averaged unless otherwise specified.",
        "",
        "## Table 1. Main LLM trajectory and clean ablations",
        "",
        "| Experiment | Setting | Precision | Recall | micro-F1 | macro-F1 | Hamming loss | Subset accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in MAIN_RESULTS:
        lines.append(
            f"| {row['experiment']} | {row['setting']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['micro_f1']:.4f} | {row['macro_f1']:.4f} | {row['hamming_loss']:.4f} | {row['subset_accuracy']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Table 2. Traditional cheminformatics baselines on the main split",
            "",
            "| Model | Subset accuracy | micro-F1 | macro-F1 | micro precision | micro recall | Hamming loss |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, subset, micro, macro, precision, recall, hamming in BASELINES:
        lines.append(f"| {model} | {subset:.4f} | {micro:.4f} | {macro:.4f} | {precision:.4f} | {recall:.4f} | {hamming:.4f} |")

    lines.extend(
        [
            "",
            "## Table 3. Scaffold split generalization",
            "",
            "| Model | Subset accuracy | micro-F1 | macro-F1 | micro precision | micro recall | Hamming loss |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, subset, micro, macro, precision, recall, hamming in SCAFFOLD:
        lines.append(f"| {model} | {subset:.4f} | {micro:.4f} | {macro:.4f} | {precision:.4f} | {recall:.4f} | {hamming:.4f} |")

    lines.extend(
        [
            "",
            "## Table 4. Rare-label effect of constrained calibration",
            "",
            "| Label | Ensemble F1 | Constrained F1 | Ensemble recall | Constrained recall | Ensemble precision | Constrained precision |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, base_f1, cal_f1, base_rec, cal_rec, base_prec, cal_prec in RARE_LABELS:
        lines.append(f"| {label} | {base_f1:.4f} | {cal_f1:.4f} | {base_rec:.4f} | {cal_rec:.4f} | {base_prec:.4f} | {cal_prec:.4f} |")

    lines.extend(
        [
            "",
            "## Table 5. Leakage audit summary",
            "",
            "| Split | Audit item | Count |",
            "|---|---|---:|",
        ]
    )
    for split, item, count in LEAKAGE:
        lines.append(f"| {split} | {item} | {count} |")

    lines.extend(
        [
            "",
            "## Table 6. Cross-LLM robustness on the main split",
            "",
            "| Model | Setting | Precision | Recall | micro-F1 | macro-F1 | Hamming loss | Subset accuracy | Parse failures |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in CROSS_LLM:
        lines.append(
            f"| {row['model']} | {row['setting']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['micro_f1']:.4f} | {row['macro_f1']:.4f} | {row['hamming_loss']:.4f} | "
            f"{row['subset_accuracy']:.4f} | {row['parse_failures']} |"
        )

    lines.extend(
        [
            "",
            "## Table 7. Qwen2.5-7B-Instruct bootstrap confidence intervals",
            "",
            "| Metric | Point | CI95 low | CI95 high |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, point, low, high in QWEN_CI:
        lines.append(f"| {metric} | {point:.4f} | {low:.4f} | {high:.4f} |")

    lines.extend(
        [
            "",
            "## Recommended claims",
            "",
            "- Zero-shot Mistral is far below LoRA-tuned models, supporting the need for task-specific SFT.",
            "- Rare-label oversampling improves macro-level performance more clearly than pictograms alone.",
            "- Pictograms are useful as weak evidence when combined with long-tail sampling and ensembling.",
            "- Qwen2.5-7B-Instruct reproduces the main LoRA workflow with zero parse failures and remains above the strongest conventional baseline, supporting cross-LLM robustness.",
            "- Constrained calibration is best interpreted as a long-tail/macro-F1 oriented post-processing step, not a universally dominant model.",
            "- Scaffold split removes CID, SMILES, and scaffold train-test overlap and should be used as the main structure-level generalization evidence.",
        ]
    )
    (ROOT / "final_tables_for_manuscript.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_references() -> None:
    text = """# Reference Seed List

Use this list as a starting point when building the formal bibliography. Verify journal style, author order, page numbers, and DOI before submission.

## Core data and cheminformatics

1. United Nations. Globally Harmonized System of Classification and Labelling of Chemicals (GHS). Use the latest Purple Book edition relevant to the manuscript.
2. Kim S, Chen J, Cheng T, Gindulyte A, He J, He S, Li Q, Shoemaker BA, Thiessen PA, Yu B, Zaslavsky L, Zhang J, Bolton EE. PubChem in 2021: new data content and improved web interfaces. Nucleic Acids Research. 2021;49(D1):D1388-D1395. doi:10.1093/nar/gkaa971.
3. Weininger D. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. Journal of Chemical Information and Computer Sciences. 1988;28(1):31-36. doi:10.1021/ci00057a005.
4. Morgan HL. The generation of a unique machine description for chemical structures: a technique developed at Chemical Abstracts Service. Journal of Chemical Documentation. 1965;5(2):107-113. doi:10.1021/c160017a018.
5. Rogers D, Hahn M. Extended-connectivity fingerprints. Journal of Chemical Information and Modeling. 2010;50(5):742-754. doi:10.1021/ci100050t.
6. Bemis GW, Murcko MA. The properties of known drugs. 1. Molecular frameworks. Journal of Medicinal Chemistry. 1996;39(15):2887-2893. doi:10.1021/jm9602928.
7. RDKit: Open-source cheminformatics software. Cite the version and Zenodo/software citation used in the project.
8. Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, Blondel M, Prettenhofer P, Weiss R, Dubourg V, Vanderplas J, Passos A, Cournapeau D, Brucher M, Perrot M, Duchesnay E. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. 2011;12:2825-2830.

## LLMs and fine-tuning

9. Jiang AQ, Sablayrolles A, Mensch A, et al. Mistral 7B. arXiv:2310.06825.
10. Hu EJ, Shen Y, Wallis P, Allen-Zhu Z, Li Y, Wang S, Wang L, Chen W. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
11. Wolf T, Debut L, Sanh V, Chaumond J, Delangue C, Moi A, Cistac P, Rault T, Louf R, Funtowicz M, Davison J, Shleifer S, von Platen P, Ma C, Jernite Y, Plu J, Xu C, Le Scao T, Gugger S, Drame M, Lhoest Q, Rush AM. Transformers: State-of-the-art natural language processing. EMNLP 2020 System Demonstrations. 2020;38-45. doi:10.18653/v1/2020.emnlp-demos.6.
12. Mangrulkar S, Gugger S, Debut L, Belkada Y, Paul S, Bossan B. PEFT: State-of-the-art parameter-efficient fine-tuning methods. Cite the software version used in the project.
13. Dettmers T, Pagnoni A, Holtzman A, Zettlemoyer L. QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.
14. Fang Y, Liang X, Zhang N, Liu K, Huang R, Chen Z, Fan X, Chen H. Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models. arXiv:2306.08018.
15. Zhang D, Liu W, Tan Q, et al. ChemLLM: A Chemical Large Language Model. arXiv:2402.06852.
16. Yu B, Baker FN, Chen Z, Ning X, Sun H. LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset. arXiv:2402.09391.

## Journal fit and reporting

17. Journal of Cheminformatics aims and scope: chemical information systems, chemical structure representations, QSAR, and data mining techniques.
18. Journal of Cheminformatics / Springer Nature author guidance for data, materials, and software availability.

## Optional safety and ecotoxicity context

19. Add references for aquatic toxicity, bioaccumulation, environmental fate, LC50/EC50/NOEC, and GHS aquatic hazard classification if environmental hazard discussion is expanded.
"""
    (ROOT / "reference_seed_list.md").write_text(text, encoding="utf-8")


def write_supplementary_outline() -> None:
    text = """# Supplementary Materials Draft

## Supplementary Note 1. Label mapping and evidence control

Describe the mapping from GHS/H-statements to the nine coarse labels. State that H-statements and full GHS evidence were used only as oracle-like upper-bound evidence and were excluded from fair predictive settings.

## Supplementary Note 2. Data leakage audit

Report CID, SMILES, and scaffold overlap for the main multilabel-stratified split and the Bemis-Murcko scaffold split.

Key result:

- Main split: CID train-test overlap = 0, SMILES train-test overlap = 22, scaffold train-test overlap = 185.
- Scaffold split: CID train-test overlap = 0, SMILES train-test overlap = 0, scaffold train-test overlap = 0.

## Supplementary Table S1. Dataset label distribution

Use `label_distribution_seed123_pictograms_rare2x.csv`.

## Supplementary Table S2. Label co-occurrence matrix

Use `test_label_cooccurrence_seed123_pictograms_rare2x.csv`.

## Supplementary Table S3. Full per-label metrics

Use `selected_per_label_metrics.csv`.

## Supplementary Table S4. Negative and exploratory experiments

Include non-H-code evidence, rubric prompt, targeted replay, environmental reviewer, label-wise routing, and scaffold ensemble3.

## Supplementary Table S5. Cross-LLM robustness

Use `cross_llm_robustness.csv` and `qwen_bootstrap_ci.csv`. Report Qwen2.5-7B-Instruct as a robustness check under the same seed123 pictograms + rare2x LoRA protocol. Qwen achieved zero parse failures and remained above the strongest conventional baseline, although it did not exceed the Mistral main backbone. Bootstrap intervals were micro-F1 0.7501 [0.7366, 0.7641] and macro-F1 0.7507 [0.7048, 0.7823].

## Supplementary Table S6. Training and inference commands

Use final command files:

- `next_jcheminformatics_experiment_commands.md`
- `final_paper_audit_and_stats_commands.md`
- `run_final_paper_audit_and_stats.sh`
- `qwen_robustness_experiment_commands.md`
- `run_qwen_main_robustness.sh`

## Supplementary Figure S1. Leakage audit summary

Use `figures/figure_5_leakage_audit.png`.

## Supplementary Figure S2. Environmental hazard error analysis

Use `figures/figure_6_environmental_error_analysis.png`.

## Supplementary Figure S3. Cross-LLM robustness

Use `figures/figure_7_cross_llm_robustness.png`.
"""
    (ROOT / "supplementary_materials_draft.md").write_text(text, encoding="utf-8")


def write_revision_notes() -> None:
    text = """# Manuscript Revision Notes

## Add to Methods: Data leakage audit

We audited potential data leakage at the CID, SMILES, and Bemis-Murcko scaffold levels. The primary multilabel-stratified split had no CID overlap between training, validation, and test sets, but contained a small number of repeated ConnectivitySMILES and substantial scaffold overlap. Therefore, we used an additional Bemis-Murcko scaffold split as the main structure-level generalization assessment. In the scaffold split, CID, SMILES, and scaffold train-test overlap were all zero.

## Add to Results: Fine-tuning necessity and clean ablations

Zero-shot Mistral performed poorly compared with LoRA-tuned models, with micro-F1 0.4175 and macro-F1 0.2763. This demonstrates that task-specific supervised fine-tuning is necessary. Clean ablations showed that no-GHS + rare2x achieved macro-F1 0.7538, while pictograms without rare2x achieved macro-F1 0.7296. The best stable main-split workflow was pictograms + rare2x + ensemble3, with micro-F1 0.7655 and hamming loss 0.1198. These results indicate that weak evidence is most useful when combined with long-tail sampling and ensembling.

## Add to Results: Scaffold ensemble note

The scaffold ensemble3 did not outperform the single scaffold model. The single Mistral LoRA scaffold model achieved micro-F1 0.7011 and macro-F1 0.5647, whereas scaffold ensemble3 achieved micro-F1 0.7000 and macro-F1 0.5563. This suggests that scaffold-level errors mainly reflect structure-level distribution shift rather than single-seed training stochasticity.

## Add to Discussion: Evidence leakage

H-statements and full GHS evidence are closely tied to label construction and should not be treated as fair predictive inputs. They are best presented as oracle-like upper bounds. GHS pictograms are weaker evidence and more appropriate for the main weak-evidence setting, but they are still GHS-derived and should not be framed as fully de novo structure-only prediction.

## Recommended figure order

1. Figure 1: Workflow overview.
2. Figure 2: Main split ablation trajectory.
3. Figure 3: Rare-label constrained calibration effect.
4. Figure 4: Scaffold split comparison.
5. Figure 5: Leakage audit summary.
6. Figure 6: Environmental hazard error analysis.
"""
    (ROOT / "draft_revision_notes.md").write_text(text, encoding="utf-8")


def plot_figures() -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"matplotlib unavailable, skipping PNG figures: {exc}")
        return

    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Figure 1: simple workflow as SVG.
    workflow_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="320" viewBox="0 0 1200 320">
<style>
.box{fill:#eef4ff;stroke:#2f5f9f;stroke-width:2;rx:14}
.txt{font-family:Arial,sans-serif;font-size:20px;fill:#15324b;text-anchor:middle}
.small{font-size:15px}
.arrow{stroke:#476b8f;stroke-width:3;marker-end:url(#arrow)}
</style>
<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#476b8f"/></marker></defs>
<rect class="box" x="30" y="80" width="160" height="110"/>
<text class="txt" x="110" y="125">PubChem</text><text class="txt small" x="110" y="155">structured fields</text>
<line class="arrow" x1="190" y1="135" x2="250" y2="135"/>
<rect class="box" x="250" y="80" width="170" height="110"/>
<text class="txt" x="335" y="120">Evidence control</text><text class="txt small" x="335" y="150">pictograms / H-code</text>
<line class="arrow" x1="420" y1="135" x2="480" y2="135"/>
<rect class="box" x="480" y="80" width="170" height="110"/>
<text class="txt" x="565" y="120">SFT prompts</text><text class="txt small" x="565" y="150">JSON labels</text>
<line class="arrow" x1="650" y1="135" x2="710" y2="135"/>
<rect class="box" x="710" y="80" width="150" height="110"/>
<text class="txt" x="785" y="120">LoRA-tuned</text><text class="txt small" x="785" y="150">Mistral</text>
<line class="arrow" x1="860" y1="135" x2="920" y2="135"/>
<rect class="box" x="920" y="80" width="160" height="110"/>
<text class="txt" x="1000" y="120">Ensemble +</text><text class="txt small" x="1000" y="150">calibration</text>
<line class="arrow" x1="1080" y1="135" x2="1130" y2="135"/>
<rect class="box" x="1130" y="80" width="50" height="110"/>
<text class="txt small" x="1155" y="125">9</text><text class="txt small" x="1155" y="150">labels</text>
</svg>
"""
    (FIG_DIR / "figure_1_workflow.svg").write_text(workflow_svg, encoding="utf-8")

    # Figure 2: main ablation.
    labels = [r["experiment"] for r in MAIN_RESULTS]
    x = np.arange(len(labels))
    width = 0.28
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width, [r["micro_f1"] for r in MAIN_RESULTS], width, label="micro-F1", color="#3568a8")
    ax.bar(x, [r["macro_f1"] for r in MAIN_RESULTS], width, label="macro-F1", color="#69a761")
    ax.bar(x + width, [r["hamming_loss"] for r in MAIN_RESULTS], width, label="hamming loss", color="#d98244")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Metric value")
    ax.set_title("Main split ablation trajectory")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_2_main_ablation.png")
    plt.close(fig)

    # Figure 3: rare-label calibration.
    labels = [r[0] for r in RARE_LABELS]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    axes[0].bar(x - 0.18, [r[1] for r in RARE_LABELS], 0.36, label="Ensemble", color="#3568a8")
    axes[0].bar(x + 0.18, [r[2] for r in RARE_LABELS], 0.36, label="Constrained", color="#69a761")
    axes[0].set_title("F1")
    axes[1].bar(x - 0.18, [r[3] for r in RARE_LABELS], 0.36, label="Ensemble", color="#3568a8")
    axes[1].bar(x + 0.18, [r[4] for r in RARE_LABELS], 0.36, label="Constrained", color="#69a761")
    axes[1].set_title("Recall")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Metric value")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Effect of constrained calibration on rare labels", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_3_rare_label_calibration.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: scaffold.
    labels = [r[0] for r in SCAFFOLD]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, [r[2] for r in SCAFFOLD], 0.4, label="micro-F1", color="#3568a8")
    ax.bar(x + 0.2, [r[3] for r in SCAFFOLD], 0.4, label="macro-F1", color="#69a761")
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("F1")
    ax.set_title("Scaffold split generalization")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_4_scaffold_comparison.png")
    plt.close(fig)

    # Figure 5: leakage audit.
    labels = [f"{split}\n{item.replace(' train-test overlap', '')}" for split, item, _ in LEAKAGE]
    counts = [count for _, _, count in LEAKAGE]
    colors = ["#d98244" if count else "#69a761" for count in counts]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(len(labels)), counts, color=colors)
    ax.set_ylabel("Overlap count")
    ax.set_title("Train-test leakage audit")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_5_leakage_audit.png")
    plt.close(fig)

    # Figure 6: environmental errors.
    labels = [r[0] for r in ENV_ERROR]
    counts = [r[1] for r in ENV_ERROR]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, counts, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
    ax.set_ylabel("Count in reviewed sample")
    ax.set_title("Environmental hazard error review")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_6_environmental_error_analysis.png")
    plt.close(fig)

    # Figure 7: cross-LLM robustness.
    labels = [r["model"] for r in CROSS_LLM]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - 0.18, [r["micro_f1"] for r in CROSS_LLM], 0.36, label="micro-F1", color="#3568a8")
    ax.bar(x + 0.18, [r["macro_f1"] for r in CROSS_LLM], 0.36, label="macro-F1", color="#69a761")
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("F1")
    ax.set_title("Cross-LLM robustness on the main split")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_7_cross_llm_robustness.png")
    plt.close(fig)


def main() -> None:
    write_csv(ROOT / "main_results_and_ablations.csv", MAIN_RESULTS, [
        "experiment", "setting", "precision", "recall", "micro_f1", "macro_f1", "hamming_loss", "subset_accuracy"
    ])
    write_csv(ROOT / "cross_llm_robustness.csv", CROSS_LLM, [
        "model", "setting", "precision", "recall", "micro_f1", "macro_f1", "hamming_loss", "subset_accuracy", "parse_failures"
    ])
    write_csv(ROOT / "qwen_bootstrap_ci.csv", QWEN_CI, [
        "metric", "point", "ci95_low", "ci95_high"
    ])
    write_csv(ROOT / "traditional_baselines_main_split.csv", BASELINES, [
        "model", "subset_accuracy", "micro_f1", "macro_f1", "micro_precision", "micro_recall", "hamming_loss"
    ])
    write_csv(ROOT / "scaffold_split_results.csv", SCAFFOLD, [
        "model", "subset_accuracy", "micro_f1", "macro_f1", "micro_precision", "micro_recall", "hamming_loss"
    ])
    write_csv(ROOT / "rare_label_calibration_effect.csv", RARE_LABELS, [
        "label", "ensemble_f1", "constrained_f1", "ensemble_recall", "constrained_recall", "ensemble_precision", "constrained_precision"
    ])
    write_csv(ROOT / "leakage_audit_summary.csv", LEAKAGE, ["split", "audit_item", "count"])
    write_csv(ROOT / "environmental_hazard_error_review.csv", ENV_ERROR, ["category", "count"])

    write_markdown_tables()
    write_references()
    write_supplementary_outline()
    write_revision_notes()
    plot_figures()
    print(f"Paper assets written to: {ROOT}")


if __name__ == "__main__":
    main()
