# ChemHazard-LLM experimental dataset card

This card documents the historical experiment datasets. The public release is **code-only**: the `data/` directory, row-level PubChem or NITE records, ID-level split manifest, and official workbook are not distributed. The counts and hashes documented here are descriptive, not a substitute for the original inputs.

## PubChem development set

- Experiment universe: 11,556 compounds retained from a 50,000-row processed export.
- Source lineage: PubChem PUG REST basic fields plus PUG-View GHS, physical-property, and toxicity fields; original scripts and the CID-based merge are documented in `data_acquisition/README.md`.
- Task: nine coarse-grained GHS hazard labels.
- Split: custom iterative multi-label stratification, seed 123; unique train/validation/test counts 9,244/1,156/1,156.
- Simple five-model comparison: no pictogram evidence or oversampling.
- Main ChemHazard-LLM setting: pictogram weak evidence and training-only rare2x oversampling; 12,990 effective training rows, with validation and test unchanged.
- Oversampling targets: `oxidizing`, `gas_under_pressure`, `cmr`, `stot`, and `environmental_hazard`.

The processed source table, JSONL splits, and per-ID split manifest exist locally but are excluded from the software release. The original collection procedure and export hash are documented in `DATA_REQUIREMENTS.md`; the exact historical API responses and original snapshot are not public here, so a fresh download is not guaranteed to reproduce the fixed split or reported scores.

## NITE Japan-GHS external set

- Official English workbook update date: 2026-09-02; local download date: 2026-09-24.
- Source rows: 3,476.
- Primary exact-full-InChIKey-disjoint test set: 1,053 compounds after excluding 2,132 overlaps against the 11,556-compound PubChem reference universe.
- Connectivity-block-disjoint sensitivity subset: 1,016 compounds.
- Unknown, blank, `-`, and `Classification not possible` cells remain unknown and are excluded from scoring through `label_mask=0`.

The model input omits NITE classification text, signal words, pictograms, and hazard statements. The official workbook, audit spreadsheets, source-audit summaries, and row-level external records remain local. The recorded workbook checksum is in `DATA_REQUIREMENTS.md`.

## Integrity and limitations

The historical PubChem and NITE input hashes are stated in `DATA_REQUIREMENTS.md`. Running `generate_data_checksums.py` on a different download may produce different hashes. The current software archive alone does not guarantee exact historical reruns. A future data release, if rights permit, must identify its source terms, version, retrieval date, schema, and checksums.
