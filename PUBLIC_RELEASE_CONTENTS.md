# Public release contents

## Intended for GitHub and the Zenodo software archive

- source code for preprocessing, training, inference, evaluation, baselines, ablations, figures, scaffold splitting, bootstrap analysis, and NITE construction;
- fixed experiment configurations and seed/threshold declarations;
- model repository identifiers, exact revisions, a sanitized public checksum manifest, and identity-capture utilities;
- aggregate data summaries, non-row-level split metadata, dataset card, and SHA-256 manifests;
- environment specifications, sanitized `chem2`/`chem` runtime captures, release documentation, and automated release checks.

## Kept locally but excluded from the public source repository

- official NITE workbook and row-level NITE-derived files pending written redistribution confirmation;
- processed PubChem tables and generated JSONL files pending source-level license review;
- the ID-level seed-123 split manifest, because the current release withholds all row-level data;
- foundation-model weights and tokenizer snapshots;
- trained LoRA adapters, checkpoints, predictions, logs, and the unsanitized machine-specific model capture.

## Recommended separate archival records

If redistribution is permitted, create a separate Zenodo dataset record for processed data and a separate versioned model/adapters record when necessary. This prevents the software DOI from ambiguously licensing third-party data or model artifacts. Link the records through Zenodo related identifiers and list all DOIs in the repository and manuscript.

The repository's MIT License applies only to original code and documentation unless a file explicitly states otherwise.
