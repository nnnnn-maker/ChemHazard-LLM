# Data requirements for the code-only release

The public GitHub repository and its Zenodo **software** archive contain code, experiment settings, aggregate summaries, source metadata, and checksums. They do **not** contain the processed PubChem table, fixed train/validation/test JSONL files, the ID-level split manifest, the official NITE workbook, or any row-level NITE-derived files. These files remain local while redistribution rights are reviewed. The MIT software license does not apply to them.

Consequently, the public package alone cannot reproduce the exact historical result tables. A reader can inspect the methods and rerun the pipeline only after obtaining the same input data under the applicable source terms. The checksums identify the historical inputs; they do not provide access to those inputs.

## PubChem development data

The historical experiment retained 11,556 compounds from a 50,000-row PubChem export. The acquisition scripts, API endpoints, merge order, label mapping, and a runnable new-snapshot collection example are documented in `data_acquisition/README.md`; the original collection scripts are included there. The historical data flow is `compounds.csv` + `ghs_data.csv` + `extended_properties.csv` -> `full_dataset.csv` -> `full_dataset_coarse_labels.csv` -> seed-123 training/validation/test JSONL. Before training-only oversampling, the fixed split contained 9,244 training, 1,156 validation, and 1,156 test compounds; the main rare2x training JSONL contained 12,990 rows. The local, excluded files are `data/pubchem/full_dataset_coarse_labels.csv`, `data/pubchem/seed123_split_manifest.csv`, and the JSONL files under `data/pubchem/seed123_simple/` and `data/pubchem/seed123_pictograms_rare2x/`.

`build_pubchem_coarse_labels.py` maps the original structured PubChem export to nine labels, and `build_pubchem_coarse_sft_dataset.py` constructs the split and training-only oversampling. The **original experiment input** `full_dataset.csv` has SHA-256 `e118c732a00c78522963e44c431f18674c1b7bc3fc9f99fb7ed0d7c6e659ea0d`; the processed coarse-label table has SHA-256 `0027bdfeee048fb81c024c60871c2c2f1637eff20966240dffb3bee0a1d65aad`. The later `full_dataset_pubchem_completed.csv` is a different version. The currently retained GHS and extended-property collection CSVs were updated after the original export, and live PubChem responses can change. Thus the collection method is now documented, but the code-only archive still cannot guarantee reconstruction of the exact historical rows or scores from public services. Do not describe the fixed seed-123 split as publicly downloadable until a separately licensed data release exists.

## NITE Japan-GHS external data

The historical official English workbook was updated on 2026-09-02 and downloaded on 2026-09-24 from:

```text
https://www.chem-info.nite.go.jp/chem/english/ghs/files/list_nite_all_e.xlsx
```

Its recorded SHA-256 is `79e5cba0a79af3e60da39d9bbcc0e13c2581ab90b11dba90582b7e7cb720777c`. The local copy and all row-level external sets are excluded from Git. The exact full-InChIKey-disjoint set contained 1,053 compounds; the connectivity-disjoint sensitivity subset contained 1,016. Rebuilding the test set requires the historical workbook, PubChem structure-resolution responses, the 11,556-compound PubChem reference set, and the construction code under `external_validation/nite_japan_ghs/`. Changes in source services may prevent bit-for-bit reconstruction without cached historical responses.

Unknown NITE classifications are represented by `label_mask=0`, not negative labels. Scores use only cells with `label_mask=1`.

See `DATASET_CARD.md`, `data/README.md`, and `data/checksums.sha256` for the recorded dataset identities. If redistribution is later cleared, publish permitted row-level files in a separately versioned data record with their own license, source attribution, checksums, and DOI; do not add them silently to the MIT-licensed software archive.
