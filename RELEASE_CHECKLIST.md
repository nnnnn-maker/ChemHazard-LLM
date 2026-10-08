# Reproducibility release checklist

Complete these items on the validated Linux/CUDA server before publishing the formal GitHub/Zenodo release. This is a code-only software package; do not include row-level data by default.

Run `python check_public_release.py --final` immediately before creating the release tag. The release must follow `GITHUB_ZENODO_RELEASE.md`.

## Already included in the code package

- Nine-label GHS mapping and PubChem preprocessing.
- Deterministic seed-123 multi-label split construction.
- Fixed split-manifest exporter with oversampling exposure counts.
- Training-only target-label oversampling.
- LoRA training, inference, and evaluation scripts.
- Main member seeds 123, 777, and 2025.
- Majority voting and validation-only constrained label-wise threshold selection.
- NITE source audit, InChIKey decontamination, and `label_mask=1` evaluation.
- Bemis-Murcko scaffold split and overlap audit.
- Sample-level and paired bootstrap scripts.
- Metric aggregation and manuscript figure/table scripts.
- Core and Chemprop environment entry files.

## Commit with the code-only release

- `data/pubchem/seed123_split_manifest.metadata.json` and `data/checksums.sha256`; these record historical identities without exposing per-compound rows. Keep `data/pubchem/seed123_split_manifest.csv` ignored.
- `reproducibility/environments/chem2/{environment.json,pip_freeze.txt,nvidia_smi.txt}` for Qwen, Mistral, and Gemma.
- `reproducibility/environments/chem/{environment.json,pip_freeze.txt,nvidia_smi.txt}` for ChatGLM and ChemLLM.
- `reproducibility/model_manifest.public.json`, sanitized from the local server capture and containing revisions and checksums for all five backbones. Keep `model_manifest.local.json` ignored because it contains machine-specific paths; a current upstream HEAD is not an acceptable substitute.
- SHA-256 checksums for historical local inputs, the official NITE workbook, and final adapters/predictions where available; checksums do not make excluded files publicly accessible.
- An explicit statement that only from-scratch retraining is supported unless the three adapters are later archived separately under their applicable terms.
- An aggregate, non-row-level validation threshold-selection audit if it is available and verified.
- Per-label metric files for all nine labels in the journal supplementary material, after verification; do not silently include incomplete working drafts in the software release.

## Release gates

- A root `LICENSE` file has been approved by the code copyright holders; its terms are not presented as applying to third-party data or model weights.
- `CITATION.cff` and `.zenodo.json` identify only Yaning Fan, use the supplied repository URL, and have matching title/license/version (`0.1.0`). Add the actual release date at release time; leave unknown affiliation, ORCID, and DOI absent.
- `VERSION`, Git tag, GitHub Release title, and Zenodo software version agree.
- NITE source files and row-level derived records are not tracked unless redistribution permission has been documented.
- PubChem-derived row-level files are not tracked unless source-level redistribution terms have been reviewed.
- The ID-level split manifest and unfinished supplementary DOCX/XLSX files are not tracked in this code-only release.
- The manuscript's Code and Data Availability statement accurately says that the software archive alone cannot recreate exact historical scores while fixed data are withheld.

- Main split unique counts are 9,244/1,156/1,156 before training replay.
- Each of the three final member prediction files contains exactly 1,156 test IDs and no parse failures.
- The reproduced validation-selected thresholds are 1 for `oxidizing`, `gas_under_pressure`, and `cmr`, and 2 for the other labels.
- The ordinary 2-of-3 ensemble is checked before constrained voting.
- No obsolete seed-42 screening output or failed Qwen all-zero output is present.
- NITE results report 1,053 gold records and score only `label_mask=1` cells.
- Scaffold audit reports parsing coverage as well as overlap; failed or empty scaffolds are not silently counted as verified non-overlap.
- Every `used_revision` field in `MODEL_MANIFEST.json` is a validated 40-character commit SHA, and the captured local model identities match the declared architecture.
- Each `experiment_access_date` is accompanied by matching-revision local-download-metadata evidence and is described as a recorded download/cache-validation date, not the first download or the later manuscript webpage-access date.
- The historical Mistral server `tokenizer_config.json` checksum and the sole `chat_template` difference are documented; the published training and inference code reproduces the historical plain-role prompt. Keep `model_snapshot_verified` as `false` because the official directory is not byte-identical, and do not publish the recovered tokenizer file without a separate redistribution review.
- The `chem2` and `chem` captures both report PyTorch, Transformers, PEFT, Accelerate, and BitsAndBytes; neither runtime is represented by an unrelated base or Chemprop environment.
