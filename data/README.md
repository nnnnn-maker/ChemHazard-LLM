# Data availability

This is a code-only release. The MIT License covers original code and documentation, not PubChem or NITE source material. No row-level experimental dataset is authorized for public redistribution at this stage.

The repository may track this README, aggregate summaries, source-audit metadata, split metadata without IDs, and `checksums.sha256`. It must not track:

- the processed PubChem CSV or either fixed JSONL split;
- `pubchem/seed123_split_manifest.csv`, because it contains per-compound IDs and training exposure counts;
- the official NITE workbook, row-level NITE CSV/JSONL files, or audit spreadsheets;
- model weights, adapters, predictions, or caches.

The historical inputs and their SHA-256 values are documented in `DATA_REQUIREMENTS.md` and `DATASET_CARD.md`. A checksum only verifies a file already obtained under its own source terms. It does not make the data publicly accessible, and the code-only archive alone is insufficient to regenerate exact paper results. If source rights are clarified, permitted datasets can be deposited separately with an explicit data license, provenance, version, and DOI.

Do not publish obsolete seed-42 screening results, the discontinued ICSC data, failed Qwen all-zero predictions, or credentials.
