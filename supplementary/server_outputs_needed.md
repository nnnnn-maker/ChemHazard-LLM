# Original server outputs needed for the remaining supplementary tables

Do not enter missing metrics manually or infer them from aggregate F1. Copy the original prediction and evaluation outputs, or regenerate evaluations from the original predictions and fixed gold files. No model retraining is required for this step.

## PubChem nine-label performance

From the `pubchem_work` project root on the experiment server, locate the final seed123 prediction file and the corresponding existing `*_per_label.csv` files:

```bash
find evaluation_llm_coarse -type f -name '*_per_label.csv' | grep -E 'seed123|split123|labelcal_constrained'
find predictions -type f -name '*labelcal_constrained*predictions.jsonl'
```

The fixed main-test gold file is `sft_coarse_jsonl_seed123_ghs_pictograms_rare2x/test.jsonl`. If the final per-label CSV is absent, use the original final prediction file with the existing evaluator:

```bash
python evaluate_pubchem_coarse_predictions.py \
  --gold sft_coarse_jsonl_seed123_ghs_pictograms_rare2x/test.jsonl \
  --pred predictions/mistral_split123_ghs_pictograms_rare2x_labelcal_constrained_ensemble3_predictions.jsonl \
  --output-dir evaluation_llm_coarse/supplementary_final \
  --prefix chemhazard_final_seed123
```

Before accepting the output, confirm 1,156 matched IDs, zero missing/extra IDs, and aggregate Micro-F1 approximately 0.7653 and Macro-F1 approximately 0.7852. The exact final prediction filename should be verified on the server. Do **not** substitute the similarly named `replace_rare2x_ensemble3_env` file merely because aggregate scores match; compare its per-ID nine-label predictions first.

For additional PubChem comparators, use their existing seed123 prediction files and the same fixed gold file. The legacy seed42 five-model results and the failed Qwen all-zero run must not be included.

## NITE nine-label performance

Find the final NITE prediction file and its mask-aware metrics JSON:

```bash
find evaluation_llm_coarse/nite_japan_ghs -type f -name '*metrics.json'
find predictions/nite_japan_ghs -type f -name '*chemhazard*predictions.jsonl'
```

If the JSON is absent, run the existing mask-aware evaluator with the fixed 1,053-row gold set:

```bash
python external_validation/nite_japan_ghs/evaluate_nite_external_predictions.py \
  --gold external_validation/nite_japan_ghs/nite_external_test_exact_inchikey_disjoint.jsonl \
  --predictions predictions/nite_japan_ghs/exact/chemhazard_llm_constrained_ensemble3_predictions.jsonl \
  --output evaluation_llm_coarse/nite_japan_ghs/exact/chemhazard_llm_supplementary_metrics.json
```

Before accepting the output, confirm 1,053 matched records, 4,630 scored cells, and the reported final aggregate Micro-F1 0.3804 and Macro-F1 0.3447. The evaluator's `per_label` object contains the required TP, FP, FN, TN and scoreable denominators.

## Transfer back to this folder

Copy only the small result files first: the PubChem `*_per_label.csv` and `*_metrics.json` for each selected model, the NITE mask-aware `*_metrics.json`, and the final prediction-file SHA-256 values. Place them in a local `supplementary/source_metrics/` directory. The raw prediction JSONL files may remain on the server if their checksums and an accessible archived release are provided, subject to the project's data-sharing decision.

Do not copy private model weights, tokens, server paths, or training logs containing credentials into the public repository.
