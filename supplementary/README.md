# Supplementary material working set

This directory separates verified local results from material that still needs the original server outputs. It is a working set, **not yet a submission-ready set of additional files**.

The spreadsheet was rendered and visually checked. The DOCX passed a structural check (three complete tables), but the automated DOCX renderer is unavailable on this Windows host; open it in Word and inspect every page before submission.

## Prepared here

- `Additional_file_1_methods_and_audits.docx` — editable English supplementary notes on the fixed split, rare-label exposure, NITE coverage, and scaffold paired bootstrap. All tables are traceable to files in this repository or the project's `paper_assets` directory.
- `Additional_file_2_verified_tables.xlsx` — machine-readable versions of the verified supplementary tables, with a provenance sheet.
- `server_outputs_needed.md` — precise list of missing outputs and commands to produce the nine-label performance table without rerunning model inference.

## Still required before submission

1. Complete per-label precision, recall, F1 and support for the fixed seed123 PubChem test set, at minimum for simple LoRA Mistral, the single-model ChemHazard-LLM configuration, and the final constrained-vote model. The final result must be recomputed or recovered from the original 1,156-row prediction file.
2. Mask-aware per-label NITE metrics for the final model and the comparator models selected for the manuscript. Keep `valid`, positive and negative counts beside each metric; unknown cells must not be treated as negative.
3. Verify the final manuscript's table and figure numbering, cite each additional file in order, and add the journal's required file description (name, format, title and content) to the manuscript.
4. Resolve data/source redistribution and software licensing before putting source spreadsheets or processed PubChem data into a public release.

The abandoned ICSC experiment is intentionally excluded. Historical seed42 five-model results and the failed all-zero Qwen run are also excluded. The current five-model comparison uses the fixed seed123 experiment only.

The journal permits spreadsheet/CSV additional tables and requires each additional file to be cited in order, described in the manuscript, and no larger than 20 MB. Recheck the current [submission guidelines](https://link.springer.com/journal/13321/submission-guidelines) when submitting.
