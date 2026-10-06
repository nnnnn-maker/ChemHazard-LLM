# NITE Japan-GHS external test set

This package is built from the official NITE latest-classification workbook dated 2026-09-02. CAS identifiers were resolved with PubChem Identifier Exchange; structure and neutral model-input properties were retrieved with PUG REST.

Use `nite_external_test_exact_inchikey_disjoint.csv` as the primary set. It contains 1,053 unique full InChIKeys after excluding every exact InChIKey in the 11,556-record seed123 experiment universe. The 1,016-record `nite_external_test_connectivity_disjoint.csv` is stricter and additionally excludes first-block (14-character) InChIKey overlap.

Gold labels are tri-state. `label_*` is 0/1 only when NITE provides enough information; `valid_*` / JSONL `label_mask` specifies scoreable cells. `-` and `Classification not possible` are unknown and must not be converted to zero. Use `evaluate_nite_external_predictions.py` for mask-aware scoring.

Prompts contain PubChem structure and physicochemical properties only. NITE categories, signal words, pictograms and H-statements are excluded from model input.

Important limitation: the historical PubChem collection did not preserve per-label contributing-source attribution. Therefore this package proves official NITE file lineage and exact-structure decontamination, but cannot retrospectively prove complete upstream provenance independence from all PubChem contributors.

The masked external set is not a balanced benchmark. Under the conservative unknown-handling rule, `acute_toxicity` and `irritant_harmful` have validated positives but no validated negatives, and only seven records are complete cases across all nine labels. Report per-label denominators and use this dataset as external robustness evidence alongside—not as a replacement for—the internal seed123 test set.
