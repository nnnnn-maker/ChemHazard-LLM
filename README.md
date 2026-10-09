# ChemHazard-LLM reproducibility package

This is a **code-only** reproducibility package for ChemHazard-LLM. It includes the experiment implementation, configurations, and data-preparation instructions. The repository's `data/` directory is excluded; row-level PubChem and NITE data (including the ID-level split manifest), model weights, LoRA adapters, and generated predictions are not distributed. The public package alone cannot reproduce the paper's exact historical scores without access to the same inputs; see `DATA_REQUIREMENTS.md`.

## Scope

The package covers:

- PubChem coarse-label construction and the fixed seed-123 multi-label split;
- zero-shot and simple-LoRA screening of five open large language models;
- the three-member ChemHazard-LLM ensemble with training seeds 123, 777, and 2025;
- validation-only constrained label-wise vote-threshold selection;
- conventional, classifier-chain, and Chemprop baselines;
- evidence and component ablations;
- Bemis-Murcko scaffold evaluation and bootstrap statistics;
- the source-audited, InChIKey-decontaminated NITE Japan-GHS external test set;
- manuscript figure and table generation.

The discontinued ICSC experiment, invalid early Qwen all-zero predictions, obsolete seed-42 five-model results, model weights, checkpoints, and generated outputs are not included.

## Start here

1. Read `MODEL_REQUIREMENTS.md` and place the five model directories under `Model/` at the revisions in `MODEL_MANIFEST.json`. The Mistral checkpoint is **Mistral-7B-Instruct-v0.1**, not the similarly named base model. On the validated server, set `CHEMHAZARD_MODEL_ROOT=../Model`; its historical Qwen and Mistral directories are misnamed, so set `CHEMHAZARD_QWEN_MODEL_PATH=../Model/Qwen-7B-Chat` and `CHEMHAZARD_MISTRAL_MODEL_PATH=../Model/Mistral-7B-v0.1` only after verifying the experiment-used file hashes.
2. Read `DATA_REQUIREMENTS.md` and `data_acquisition/README.md` for the historical PubChem acquisition, merge, mapping, and split procedure. Obtain required source inputs and the official NITE workbook under their respective source terms. Neither the fixed experimental data nor the ID-level split manifest is distributed in this code-only repository; fresh API collection is not guaranteed to recreate the paper's exact data snapshot.
3. Create the core environment from `environment.yml` or `requirements.txt`. Use the separate `requirements-chemprop.txt` environment for Chemprop.
4. Capture both LLM runtimes described in `RUNTIME_ENVIRONMENTS.json`, then run `python verify_reproduction_setup.py`.
5. Follow `paper_reproducibility_commands.md` in order.

On the validated server, run `capture_model_manifest.py --hash-weights` to archive checkpoint identities and checksums. The recorded Mistral weights match the Instruct revision. The recovered experiment `tokenizer_config.json` matches the server hash and differs from the official snapshot only by omitting `chat_template`. This package therefore forces the historical plain `SYSTEM`/`USER`/`ASSISTANT` prompt for Mistral training and inference, even when the official tokenizer supplies a chat template. The recovered file is retained locally for audit and ignored by Git; do not claim bit-for-bit directory identity. Model weights are linked to their official providers rather than copied into this repository. Exact reproduction of the reported final result additionally requires the three versioned LoRA adapters for seeds 123, 777, and 2025; otherwise the pipeline retrains them from scratch.

`reproducibility/model_manifest.public.json` contains the sanitized server-captured model hashes without private filesystem paths. Regenerate it with `capture_model_manifest.py --hash-weights --public-output reproducibility/model_manifest.public.json` after any verified model-file change. The original `model_manifest.local.json` remains ignored and should not be uploaded.

The five LLMs use two validated environments rather than one: `chem2` for Qwen, Mistral, and Gemma; `chem` for ChatGLM and ChemLLM. Capture them separately so one environment report does not overwrite the other:

```bash
# Run after activating chem2
python capture_environment.py --name chem2

# Run after activating chem
python capture_environment.py --name chem
```

Before creating a public release, complete `RELEASE_CHECKLIST.md` on the validated Linux/CUDA server. The historical dataset identity and masking policy are summarized in `DATASET_CARD.md`; this is not a data release.
The current readiness and unresolved publication items are recorded in `RELEASE_STATUS.md`.

## GitHub and archival release

The public source repository is intended to be released with a semantic version tag and archived through the GitHub-Zenodo integration. Follow `GITHUB_ZENODO_RELEASE.md` and run `python check_public_release.py --final` before creating a release tag. `PUBLIC_RELEASE_CONTENTS.md` distinguishes source files from data and model artifacts that require separate licensing or archival decisions.

The sole named author is Yaning Fan; `CITATION.cff` and `.zenodo.json` use the supplied GitHub repository URL. No ORCID, affiliation, release date, final version, or Zenodo DOI has been invented. Add the final version/date at release time and the DOI after Zenodo registration.

## License

Original source code and documentation in this repository are released under the MIT License; see `LICENSE` (copyright 2026 Yaning Fan). This license does not grant rights to third-party PubChem or NITE source material, foundation-model weights, tokenizers, or other externally licensed assets. Review each data and model source separately before redistribution.

All shell commands assume Linux and that the current directory is the root of this package. The LLM experiments require an NVIDIA CUDA environment. Chemprop should be installed in a separate Python 3.11 environment.

For the main ChemHazard-LLM pipeline, the shorter staged entry point is:

```bash
bash run_chemhazard_main_reproduction.sh prepare
bash run_chemhazard_main_reproduction.sh train 123
bash run_chemhazard_main_reproduction.sh train 777
bash run_chemhazard_main_reproduction.sh train 2025
bash run_chemhazard_main_reproduction.sh infer 123
bash run_chemhazard_main_reproduction.sh infer 777
bash run_chemhazard_main_reproduction.sh infer 2025
bash run_chemhazard_main_reproduction.sh ensemble
bash run_chemhazard_main_reproduction.sh evaluate
```

The authoritative machine-readable settings are in `configs/main_experiment.json`.

## Fixed experimental identities

- Data split: multi-label stratified, split seed 123.
- Five-model simple-LoRA comparison: training seed 123.
- ChemHazard-LLM ensemble members: training seeds 123, 777, and 2025 on the same split-123 data.
- Final relaxed vote thresholds: 1 vote for `oxidizing`, `gas_under_pressure`, and `cmr`; 2 votes for the other six labels. Threshold selection must be performed on validation predictions only.
- Primary NITE test set: 1,053 compounds, exact full-InChIKey disjoint from all 11,556 PubChem experiment compounds, evaluated only where `label_mask=1`.

The five target-label oversampling labels are `oxidizing`, `gas_under_pressure`, `cmr`, `stot`, and `environmental_hazard`. This set is intentionally different from the three labels whose final ensemble vote threshold was relaxed.

## Requirement-to-file map

| Reproducibility requirement | Included implementation |
|---|---|
| Nine-label mapping and preprocessing | `build_pubchem_coarse_labels.py`, `build_pubchem_coarse_sft_dataset.py` |
| Original PubChem acquisition and CID merge | `data_acquisition/` (four historical scripts and a provenance guide) |
| Fixed seed-123 split or split indices | deterministic builder plus `export_fixed_split_manifest.py`; the generated per-ID manifest is local-only pending a data-sharing decision |
| Target-label oversampling | `build_pubchem_coarse_sft_dataset.py` and `configs/main_experiment.json` |
| LoRA training and hyperparameters | `train_pubchem_coarse_lora_sft.py`, `run_chemhazard_main_reproduction.sh` |
| Seeds 123/777/2025 | `configs/main_experiment.json`, `run_chemhazard_main_reproduction.sh` |
| Ensemble and label-wise voting | `combine_multilabel_votes.py`, `calibrate_multilabel_vote_thresholds.py` |
| NITE label-mask evaluation | `external_validation/nite_japan_ghs/evaluate_nite_external_predictions.py` |
| Scaffold split and bootstrap | `build_pubchem_coarse_scaffold_sft_dataset.py`, `bootstrap_ci_pubchem.py`, `paired_bootstrap_delta.py` |
| Main metrics and tables | `evaluate_pubchem_coarse_predictions.py`, `summarize_pubchem_coarse_llm_results.py` (CSV/Markdown summaries) |
| Environment specification | `requirements.txt`, `environment.yml`, `requirements-chemprop.txt`, `capture_environment.py` |
| Model-to-runtime mapping | `RUNTIME_ENVIRONMENTS.json` |
| Five backbone identities and revisions | `MODEL_MANIFEST.json`, `MODEL_REQUIREMENTS.md`, `capture_model_manifest.py` |

## Reproducibility rule

Do not mix results from different data splits, failed parsing runs, or earlier local model aliases. A run is valid only when its input IDs, label order, parse-failure count, model identity, seed, and output checksum are recorded.
