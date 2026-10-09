# Release status (2026-10-09)

This folder is prepared as a **development-stage, code-only** software package. The sole named software author is Yaning Fan; the intended GitHub repository is `https://github.com/nnnnn-maker/ChemHazard-LLM`. No GitHub Release or Zenodo DOI has been created by this package-preparation step.

The first code-only release version is set to `0.1.0` consistently in `VERSION`, `CITATION.cff`, and `.zenodo.json`. The development audit passes. The strict release audit intentionally has one remaining error because the actual release date has not yet been set in `CITATION.cff`. Add the date only on the publication day; the planned Git tag and GitHub Release are `v0.1.0`.

The source repository has been initialized and pushed to GitHub. The `data/` directory is now excluded from the next code-only commit. Locally retained PubChem CSV/JSONL files, the NITE workbook and derived records, and the recovered tokenizer configuration must remain untracked. No formal GitHub Release or Zenodo DOI has been created.

The public code repository must not include the `data/` directory, row-level PubChem or NITE files, the ID-level split manifest, the official NITE workbook, foundation-model weights, trained adapters, or generated predictions. The PubChem collection and merge procedure and original export hash are documented in `DATA_REQUIREMENTS.md`; the original API response snapshot is not distributed. The software archive alone cannot reproduce the exact paper tables. Future data-redistribution permissions remain unresolved; do not claim a public exact-reconstruction route until these are established.

Before manuscript submission or archival release, verify the selected per-label supplementary metrics, decide whether a separately licensed data/adapters archive is possible, and make the manuscript's Code and Data Availability statement match the actual access conditions. A final GitHub/Zenodo software release must pass `python check_public_release.py --final` and a staged-file review.
