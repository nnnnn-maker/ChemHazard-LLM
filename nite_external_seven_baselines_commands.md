# NITE external evaluation of seven cheminformatics baseline configurations

These scripts reproduce the seven baseline configurations from Table 4 as an
external-transfer experiment. Six sklearn models are fitted on the frozen
PubChem seed123 training split. Chemprop reuses the frozen checkpoint trained
for Table 4. NITE labels are never used for fitting, preprocessing, model
selection, or threshold selection.

## Files to upload to the project root

```text
run_nite_external_seven_baselines.sh
run_nite_external_sklearn_baselines.py
prepare_nite_chemprop_predict.py
convert_nite_chemprop_predictions.py
collect_nite_external_baseline_results.py
```

The server must already contain:

```text
external_validation/nite_japan_ghs/nite_external_test_exact_inchikey_disjoint.jsonl
external_validation/nite_japan_ghs/evaluate_nite_external_predictions.py
processed_coarse/full_dataset_coarse_labels.csv
sft_coarse_jsonl_seed123_ghs_pictograms_rare2x/train.jsonl
run_pubchem_split_baselines.py
run_pubchem_classifier_chain_baseline.py
checkpoints_chemprop/chemprop_seed123_multitask/
```

## Step 1: six sklearn baselines

Run in the existing chem2/sklearn/RDKit environment:

```bash
cd /path/to/ChemHazard_LLM_reproducibility
bash run_nite_external_seven_baselines.sh sklearn
```

This runs the following frozen PubChem-trained configurations:

```text
morgan_physchem_rf
physchem_rf
morgan_rf
physchem_logreg
morgan_logreg
morgan_physchem_classifier_chain_rf
```

## Step 2: Chemprop

Activate the same Chemprop 2.3 environment used for the internal Table 4 run:

```bash
conda activate chemprop23
cd /path/to/ChemHazard_LLM_reproducibility
bash run_nite_external_seven_baselines.sh chemprop
```

The command must reuse this frozen PubChem checkpoint by default:

```text
checkpoints_chemprop/chemprop_seed123_multitask
```

Do not retrain or tune Chemprop on NITE. The fixed probability threshold is
0.5, matching the internal experiment.

## Step 3: collect the seven results

```bash
bash run_nite_external_seven_baselines.sh collect
```

Final result files:

```text
evaluation_llm_coarse/nite_japan_ghs/exact/baselines/seven_baselines_summary.csv
evaluation_llm_coarse/nite_japan_ghs/exact/baselines/seven_baselines_summary.json
```

Every model must satisfy:

```text
gold_records=1053
prediction_records=1053
matched_records=1053
missing_gold_ids_in_predictions=0
extra_prediction_ids=0
scored_label_cells=4630
missing_prediction_cells=0
```

## Strict connectivity-disjoint sensitivity analysis

After completing the primary experiment, run the strict 1,016-record subset:

```bash
NITE_INPUT=external_validation/nite_japan_ghs/nite_external_test_connectivity_disjoint.jsonl \
NITE_TAG=connectivity \
bash run_nite_external_seven_baselines.sh sklearn

conda activate chemprop23
NITE_INPUT=external_validation/nite_japan_ghs/nite_external_test_connectivity_disjoint.jsonl \
NITE_TAG=connectivity \
bash run_nite_external_seven_baselines.sh chemprop

NITE_TAG=connectivity \
bash run_nite_external_seven_baselines.sh collect
```

Use the exact-InChIKey-disjoint result as the primary external table and the
connectivity-disjoint result as a sensitivity analysis. NITE metrics are
conditional on `label_mask=1` cells and are not directly interchangeable with
the fully observed PubChem metrics.
