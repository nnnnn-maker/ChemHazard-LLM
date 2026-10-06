# ChemHazard-LLM component ablation

This workflow adds a second ablation table without replacing the existing
evidence-source ablation. It uses only the valid seed123 experiment family and
the confirmed ensemble member seeds 123, 777, and 2025.

## 1. Audit existing assets first

```bash
cd /path/to/ChemHazard_LLM_reproducibility
bash component_ablation/run_chemhazard_component_ablation.sh audit
```

Inspect:

```text
analysis_outputs/chemhazard_component_ablation/component_ablation_audit.json
analysis_outputs/chemhazard_component_ablation/component_ablation_table.csv
analysis_outputs/chemhazard_component_ablation/component_effect_deltas.csv
```

The audit evaluates every prediction file that already exists and lists only
missing or invalid runs. It also verifies that all component variants use the
same 1,156 test compounds and identical gold labels.

## 2. Run only missing components

Only use the modes reported as missing by the audit:

```bash
bash component_ablation/run_chemhazard_component_ablation.sh zeroshot
bash component_ablation/run_chemhazard_component_ablation.sh noghs_rare2x
bash component_ablation/run_chemhazard_component_ablation.sh pictogram_norare
```

Each command refuses to overwrite an existing prediction. Existing adapters
are reused; a missing adapter is trained using the same 4-bit BF16 configuration
as the final Mistral members.

After completing missing runs:

```bash
bash component_ablation/run_chemhazard_component_ablation.sh audit
```

## 3. Paired bootstrap component deltas

After every row is complete:

```bash
bash component_ablation/run_chemhazard_component_ablation.sh bootstrap
```

The controlled contrasts are:

```text
LoRA:                  base zero-shot+pictogram -> LoRA+pictogram without rare2x
Pictogram:             no-pictogram+rare2x -> pictogram+rare2x
Rare2x:                pictogram without rare2x -> pictogram+rare2x
Ensemble:              seed123 single -> three-seed 2/3 majority vote
Constrained threshold: majority vote -> label-wise constrained vote
```

The final threshold must remain frozen:

```text
oxidizing, gas_under_pressure, cmr: 1/3
all other labels: 2/3
```

Do not use the invalid old seed42 five-model results or the early all-zero Qwen
output. Do not retune thresholds on the test set or on NITE.
