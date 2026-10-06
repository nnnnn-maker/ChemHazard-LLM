# Historical PubChem data acquisition and processing

This directory contains copies of the four original collection/merge scripts used for the PubChem export. Their original comments and terminal messages are preserved as part of the historical implementation. The public repository does **not** include their row-level outputs.

## Data lineage

1. `pubchem_scraper.py` requests basic compound properties for CID 1–50,000 through PubChem PUG REST and writes `compounds.csv`.
2. `ghs_scraper.py` requests the PUG-View `GHS Classification` section for the CIDs in `compounds.csv` and writes `ghs_data.csv`, including H-statements, pictograms, and signal words where returned.
3. `extended_scraper.py` requests PUG-View compound records for the same CIDs and writes `extended_properties.csv`, including available physical-property and toxicity fields.
4. `merge_dataset.py` left-joins the three CSV files on `CID`, retaining the base compound order, and writes `full_dataset.csv` (50,000 rows and 35 columns in the historical export).
5. The repository-root `build_pubchem_coarse_labels.py` extracts H-codes from `H_Statements` and maps them to nine binary coarse labels. Its `full_dataset_coarse_labels.csv` output retains all 50,000 rows; 11,565 have at least one extracted H-code, and 11,556 have at least one of the nine target labels.
6. The repository-root `build_pubchem_coarse_sft_dataset.py` retains those 11,556 labeled compounds, creates the fixed seed-123 train/validation/test split (9,244/1,156/1,156), formats model inputs, and applies rare2x repetition **only to the training split** for the main setting. The five-model simple setting does not use pictograms or oversampling.

The methods are implemented in the named scripts, not in the unpublished CSV/JSONL files. The mapping rules are in `build_pubchem_coarse_labels.py`; the split and prompt rules are in `build_pubchem_coarse_sft_dataset.py`.

## How to run the collection pipeline on a new snapshot

These commands are an example for creating a **new** dataset snapshot, not a claim that a present-day PubChem response will reproduce the historical file byte for byte. Run them from the repository root after installing `requirements-core.txt`. Use a clean output directory; the collectors maintain progress files and can resume previous runs.

```bash
python data_acquisition/pubchem_scraper.py --start 1 --end 50000 --output pubchem_collection
python data_acquisition/ghs_scraper.py --input pubchem_collection/compounds.csv --output pubchem_collection
python data_acquisition/extended_scraper.py --input pubchem_collection/compounds.csv --output pubchem_collection
python data_acquisition/merge_dataset.py --data_dir pubchem_collection --output full_dataset.csv
python build_pubchem_coarse_labels.py \
  --input pubchem_collection/full_dataset.csv \
  --output processed_coarse/full_dataset_coarse_labels.csv \
  --summary processed_coarse/full_dataset_coarse_labels_summary.json
COARSE_INPUT=processed_coarse/full_dataset_coarse_labels.csv bash run_chemhazard_main_reproduction.sh prepare
```

The historical experiment used the **original** `full_dataset.csv`, SHA-256 `e118c732a00c78522963e44c431f18674c1b7bc3fc9f99fb7ed0d7c6e659ea0d`. The historical processed table has SHA-256 `0027bdfeee048fb81c024c60871c2c2f1637eff20966240dffb3bee0a1d65aad`. Compare hashes before treating a local input as the paper's snapshot. The later `full_dataset_pubchem_completed.csv` is a different version and must not silently replace the historical source.

The locally retained April 2026 `ghs_data.csv` and `extended_properties.csv` were updated after the original export was created. Consequently, merging those currently available collection files is **not** an exact reconstruction of the paper's original `full_dataset.csv`. Live PubChem records can also change. The repository records the historical input identities and processing method but does not provide the original row-level source snapshot or guarantee reconstruction of the exact published scores from live APIs alone.

## Missing-data and provenance limitations

A failed or timed-out API request is not evidence that a chemical has no hazard or property. The collectors store progress/failure information, but the historical merged export does not fully distinguish every missing field's cause. In particular, a missing GHS response must not be interpreted as a verified negative GHS classification. The downstream nine-label experiment includes only rows with at least one mapped positive label; this selection does not establish verified negatives for every other hazard label.

PubChem aggregates contributed records. The historical export did not preserve per-label contributing-source attribution or a complete request-response archive. Re-running the scripts documents a comparable collection procedure, not independently verified identity with the historical snapshot. Follow PubChem's current service terms and rate guidance when collecting a new snapshot; record the collection date, file hashes, API failures, and any deviations.
