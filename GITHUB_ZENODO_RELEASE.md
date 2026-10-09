# GitHub and Zenodo release procedure

This repository is designed for a versioned GitHub **code-only** release that is archived by Zenodo. The GitHub repository supplied by the author is `https://github.com/nnnnn-maker/ChemHazard-LLM`. Development commits may be uploaded after inspecting the tracked files; do not create a formal GitHub Release or Zenodo software archive until the final-release checks and manuscript availability statement are complete.

## 1. Complete release metadata

1. Confirm that the root `LICENSE` (MIT, copyright 2026 Yaning Fan) applies to all original source files being released; keep third-party data and model permissions separate.
2. Review the existing `CITATION.cff` and `.zenodo.json`. They identify only Yaning Fan as author/creator and use the supplied repository URL. Do not invent an affiliation, ORCID, historical access date, release date, or DOI.
3. The first code-only release version is `0.1.0` in `VERSION`, `CITATION.cff`, and `.zenodo.json`. On the actual publication day, add that date to `CITATION.cff` and move the planned changelog entry into a dated release section. If the publication is postponed, use the actual later date; do not invent one in advance.
4. Recheck the recorded `used_revision` values against the validated server. The five `experiment_access_date` values are now backed by revision-matching local download metadata and mean recorded download/cache-validation dates, not first-download dates; preserve their evidence fields and do not substitute manuscript webpage-access dates. The recovered Mistral tokenizer configuration lacks only the official snapshot's `chat_template`; retain the explicit historical plain-role prompt in training and inference. Do not mark the full model directory byte-identical.
5. Capture both runtime environments declared in `RUNTIME_ENVIRONMENTS.json`: `chem2` for Qwen/Mistral/Gemma and `chem` for ChatGLM/ChemLLM.

When both `.zenodo.json` and `CITATION.cff` are present, Zenodo uses `.zenodo.json` for GitHub-release metadata. `CITATION.cff` remains useful for GitHub's citation panel and must therefore remain consistent.

The DOI does not yet exist before the first GitHub-integrated Zenodo archive. For the first release, use the GitHub repository URL in `CITATION.cff`; add the registered DOI after Zenodo completes the archive.

## 2. Keep restricted and large artifacts out of Git

The repository should contain source code, configurations, and reconstruction instructions. The entire local `data/` directory is intentionally ignored pending a redistribution decision, including the following files:

- the official NITE workbook and row-level NITE-derived CSV/JSONL/XLSX files;
- the processed PubChem table, generated JSONL splits, and per-compound split manifest;
- foundation-model weights, LoRA adapters, checkpoints, predictions, logs, and the unsanitized local model capture. The sanitized `chem2` and `chem` runtime captures and `reproducibility/model_manifest.public.json` are intended for Git.

Do not use Git LFS to bypass a licensing restriction. The current software archive does not itself provide the exact historical experimental data. If redistribution is later confirmed, publish permitted research artifacts in a separate versioned Zenodo dataset record and connect it to the software record with a related identifier. Record its DOI in `DATA_REQUIREMENTS.md` and `DATASET_CARD.md`.

## 3. Run release checks

From the repository root:

```bash
python -m compileall -q .
python verify_reproduction_setup.py
python check_public_release.py
```

After completing the license, author metadata, model revisions, and exact release version, run the strict gate:

```bash
python check_public_release.py --final
```

For a development commit, the ordinary audit and a staged-file review are required; the strict gate is required before the first formal release tag. Review `git status`, then inspect the staged file list with `git diff --cached --name-only` before pushing. Confirm that no row-level data or model files are staged.

## 4. Connect GitHub and Zenodo

1. Use the existing GitHub repository `https://github.com/nnnnn-maker/ChemHazard-LLM`. If it was initialized remotely with a README or license, inspect and reconcile that history before pushing.
2. Once the public-file audit and staged-file review pass, initialize and push this prepared folder if it is still an empty repository:

```bash
git init
git branch -M main
git add .
git status --short
git commit -m "Prepare ChemHazard-LLM reproducibility release"
git remote add origin https://github.com/nnnnn-maker/ChemHazard-LLM.git
git push -u origin main
```

Inspect `git status` before committing and confirm that model weights, NITE source files, row-level restricted data, checkpoints, and caches are absent.

3. Make the repository public before Zenodo archiving. Sign in to Zenodo, connect the same GitHub account, open the GitHub integration page, synchronize repositories, and enable this repository.
4. Optionally test metadata and release behavior first with Zenodo Sandbox.
5. Only after the final release gate passes, create and push an annotated Git tag from the validated commit, replacing the example version with the chosen release version:

```bash
git tag -a v0.1.0 -m "ChemHazard-LLM reproducibility release v0.1.0"
git push origin v0.1.0
```

6. Create a GitHub Release from that exact tag.
7. Wait for Zenodo to ingest the release, then check the Zenodo record metadata, files, creators, license, version, and archival status.
8. Record both identifiers:
   - the version DOI for the exact software release;
   - the concept DOI, which represents all versions and is normally preferred when citing the evolving software project.

## 5. Add the DOI back to the repository

After Zenodo registers the DOI:

1. update `CITATION.cff` with the DOI;
2. add the Zenodo DOI badge and citation text to `README.md`;
3. add the software DOI to the manuscript Data and Code Availability section;
4. create a small follow-up release only if a new archived software version is intended. Do not silently replace files in an already cited release.

For the manuscript, cite the version DOI when referring to the exact archived code used for the paper. The concept DOI may additionally be supplied as the persistent project-level identifier.
