# Code manifest

## Data preparation

- `data_acquisition/`: original PubChem PUG REST/PUG-View collectors, CID-based merge script, and data-lineage guide; no row-level outputs are included.
- `build_pubchem_coarse_labels.py`: map source annotations to nine coarse hazard labels.
- `build_pubchem_coarse_sft_dataset.py`: create the fixed multi-label train/validation/test JSONL files and training-only rare-label oversampling.
- `export_fixed_split_manifest.py`: export unique IDs, split assignments, replay exposure counts, and JSONL checksums.
- `build_pubchem_coarse_scaffold_sft_dataset.py`: create the Bemis-Murcko scaffold split.
- `audit_dataset_splits.py`: report CID, SMILES, and Bemis-Murcko scaffold overlap.
- `build_clean_evidence_ablation_sft.py`: construct controlled evidence-ablation variants.

## LLM training, inference, and evaluation

- `train_pubchem_coarse_lora_sft.py`: 4-bit LoRA supervised fine-tuning.
- `run_pubchem_coarse_inference.py`: zero-shot or adapter-based nine-label inference.
- `evaluate_pubchem_coarse_predictions.py`: dense PubChem multi-label evaluation.
- `run_five_llm_seed123_screening.sh`: fixed-split five-model zero-shot and simple-LoRA comparison.
- `run_chemhazard_main_reproduction.sh`: staged end-to-end entry point for the three-member main model.
- `combine_multilabel_votes.py`: majority-vote ensemble.
- `calibrate_multilabel_vote_thresholds.py`: validation-only rare-label and constrained vote-threshold selection.
- `summarize_pubchem_coarse_llm_results.py`: aggregate evaluation files.

## Baselines

- `run_pubchem_split_baselines.py`: five frozen one-vs-rest conventional baselines.
- `run_pubchem_classifier_chain_baseline.py`: Morgan plus physicochemical classifier chain.
- `prepare_chemprop_multitask_csv.py`: prepare Chemprop multitask data.
- `convert_chemprop_predictions.py`: convert Chemprop probabilities to the common JSONL format.

## Ablation, statistics, and figures

- `component_ablation/`: A0-A5 component-ablation runner and audit.
- `bootstrap_ci_pubchem.py`: bootstrap confidence intervals.
- `paired_bootstrap_delta.py`: paired bootstrap model comparison.
- `plot_fig2_dataset_characterization.py`: label-frequency and label-count panels.
- `plot_fig3_bar_charts.py`: LLM performance panels.
- `plot_fig4_ablation_charts.py`: evidence-ablation panels.
- `plot_fig6_component_ablation.py`: A0-A5 component comparison.
- `paper_figures/`: framework figure and table/figure asset generation.

## NITE external validation

- `external_validation/nite_japan_ghs/build_nite_external_testset.py`: source-audited, InChIKey-decontaminated test-set construction.
- `external_validation/nite_japan_ghs/evaluate_nite_external_predictions.py`: mask-aware external evaluation.
- `run_nite_external_one_model.sh`: five-model zero-shot and task-adapted inference.
- `run_nite_external_chemhazard.sh`: fixed three-member ChemHazard-LLM external inference.
- `run_nite_external_sklearn_baselines.py`: conventional and classifier-chain external baselines.
- `prepare_nite_chemprop_predict.py` and `convert_nite_chemprop_predictions.py`: Chemprop external prediction conversion.
- `run_nite_external_seven_baselines.sh`: baseline dispatcher.
- `collect_nite_external_baseline_results.py`: aggregate external baseline results.

## Reproducibility utilities

- `paper_reproducibility_commands.md`: complete command sequence.
- `verify_reproduction_setup.py`: preflight validation of code, model identities, and input files.
- `MODEL_MANIFEST.json`: official repository IDs, expected model identities, access notes, and publication-time revision fields for all five backbones.
- `capture_model_manifest.py`: capture exact revisions, local identities, and SHA-256 checksums from the validated model installation.
- `capture_environment.py`: export runtime versions and GPU metadata.
- `RUNTIME_ENVIRONMENTS.json`: map each backbone to its validated named runtime (`chem2` or `chem`) and declare required packages.
- `generate_data_checksums.py`: write a deterministic SHA-256 manifest for local experiment data files, including files not redistributed.
- `MODEL_REQUIREMENTS.md` and `DATA_REQUIREMENTS.md`: exact external dependencies.
- `configs/main_experiment.json`: machine-readable labels, seeds, hyperparameters, oversampling policy, thresholds, and expected headline metrics.
- `requirements.txt`, `environment.yml`, and `requirements-chemprop.txt`: environment entry points.
- `RELEASE_CHECKLIST.md`: files and checks that must be archived from the validated server before publication.
- `DATASET_CARD.md`: historical PubChem/NITE versions, split identities, masking policy, and code-only release limitations.
- `GITHUB_ZENODO_RELEASE.md`: GitHub tagging, Zenodo integration, DOI, and post-archive citation workflow.
- `PUBLIC_RELEASE_CONTENTS.md`: public source-code scope and separately archived or restricted artifacts.
- `check_public_release.py`: detect credentials, local paths, model weights, oversized files, unresolved model revisions, and incomplete release metadata.
- `CITATION.cff` and `.zenodo.json`: sole-author GitHub/Zenodo metadata; templates are retained only as examples.
