from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


MODELS = {
    "Qwen2.5-7B-Instruct": ("qwen2", "Qwen2ForCausalLM"),
    "Mistral-7B-Instruct-v0.1": ("mistral", "MistralForCausalLM"),
    "gemma-2-9b-it": ("gemma2", "Gemma2ForCausalLM"),
    "chatglm3-6b": ("chatglm", "ChatGLMModel"),
    "ChemLLM-7B-Chat": ("internlm", "InternLM2ForCausalLM"),
}

REQUIRED_CODE = [
    "build_pubchem_coarse_labels.py",
    "build_pubchem_coarse_sft_dataset.py",
    "export_fixed_split_manifest.py",
    "generate_data_checksums.py",
    "build_pubchem_coarse_scaffold_sft_dataset.py",
    "audit_dataset_splits.py",
    "train_pubchem_coarse_lora_sft.py",
    "run_pubchem_coarse_inference.py",
    "evaluate_pubchem_coarse_predictions.py",
    "combine_multilabel_votes.py",
    "calibrate_multilabel_vote_thresholds.py",
    "run_five_llm_seed123_screening.sh",
    "run_chemhazard_main_reproduction.sh",
    "run_pubchem_split_baselines.py",
    "run_pubchem_classifier_chain_baseline.py",
    "prepare_chemprop_multitask_csv.py",
    "convert_chemprop_predictions.py",
    "external_validation/nite_japan_ghs/build_nite_external_testset.py",
    "external_validation/nite_japan_ghs/evaluate_nite_external_predictions.py",
    "configs/main_experiment.json",
    "requirements.txt",
    "environment.yml",
    "DATASET_CARD.md",
    "MODEL_MANIFEST.json",
    "capture_model_manifest.py",
    "RUNTIME_ENVIRONMENTS.json",
    "check_public_release.py",
    "GITHUB_ZENODO_RELEASE.md",
    "PUBLIC_RELEASE_CONTENTS.md",
    ".gitignore",
    ".gitattributes",
]

EXPECTED_SEEDS = [123, 777, 2025]
MISTRAL_FIRST_WEIGHT_SHA256 = "a464228d9c9427bf035cfd5a2e18bf0494d7231bfacc478ec5fc9f57a612d051"
MISTRAL_SERVER_TOKENIZER_CONFIG_SHA256 = "ddb008229511e51607002ffe28925001c4a9ca4177dc4de3a655d085cc610b99"
MISTRAL_SERVER_TOKENIZER_CONTENT_SHA256 = "d2391aeae3cb5625e6f2d168d600722f25d4984c7ff8ad85fcdcc9497ecd05cd"
EXPECTED_THRESHOLDS = {
    "flammable": 2,
    "oxidizing": 1,
    "gas_under_pressure": 1,
    "corrosive": 2,
    "acute_toxicity": 2,
    "irritant_harmful": 2,
    "cmr": 1,
    "stot": 2,
    "environmental_hazard": 2,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_content_sha256(path: Path) -> str:
    config = json.loads(path.read_text(encoding="utf-8"))
    # The official Instruct snapshot adds only chat_template; the historical
    # plain-role prompt is selected explicitly by the training/inference code.
    config.pop("chat_template", None)
    normalized = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument(
        "--verify-mistral",
        action="store_true",
        help="Verify the Mistral Instruct weight fingerprint; report tokenizer differences.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    model_root = Path(
        os.environ.get(
            "CHEMHAZARD_MODEL_ROOT",
            str(root.parent / "Model" if (root.parent / "Model").is_dir() else root / "Model"),
        )
    )
    errors: list[str] = []
    warnings: list[str] = []
    for relative in REQUIRED_CODE:
        if not (root / relative).is_file():
            errors.append(f"missing code: {relative}")

    config_path = root / "configs/main_experiment.json"
    if config_path.is_file():
        experiment = json.loads(config_path.read_text(encoding="utf-8"))
        if experiment.get("training", {}).get("member_seeds") != EXPECTED_SEEDS:
            errors.append("main experiment member seeds are not 123/777/2025")
        thresholds = experiment.get("ensemble", {}).get("reported_thresholds")
        if thresholds != EXPECTED_THRESHOLDS:
            errors.append("main experiment label-wise thresholds do not match the reported run")

    for directory, (model_type, architecture) in MODELS.items():
        override_name = {
            "Qwen2.5-7B-Instruct": "CHEMHAZARD_QWEN_MODEL_PATH",
            "Mistral-7B-Instruct-v0.1": "CHEMHAZARD_MISTRAL_MODEL_PATH",
        }.get(directory)
        model_path = (
            Path(os.environ[override_name])
            if override_name and os.environ.get(override_name)
            else model_root / directory
        )
        config_path = model_path / "config.json"
        if not config_path.is_file():
            if args.require_models or (args.verify_mistral and directory == "Mistral-7B-Instruct-v0.1"):
                errors.append(f"missing model config: {config_path}")
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        architectures = config.get("architectures") or []
        if config.get("model_type") != model_type or architecture not in architectures:
            errors.append(
                f"model identity mismatch for {directory}: "
                f"model_type={config.get('model_type')}, architectures={architectures}"
            )
        if args.verify_mistral and directory == "Mistral-7B-Instruct-v0.1":
            first_weight = model_path / "model-00001-of-00002.safetensors"
            if not first_weight.is_file():
                errors.append(f"missing Mistral Instruct weight shard: {first_weight}")
            elif sha256(first_weight) != MISTRAL_FIRST_WEIGHT_SHA256:
                errors.append(
                    "Mistral first weight shard does not match the validated "
                    "Mistral-7B-Instruct-v0.1 experiment checkpoint"
                )
            tokenizer_config = model_path / "tokenizer_config.json"
            if not tokenizer_config.is_file():
                errors.append(f"missing Mistral tokenizer configuration: {tokenizer_config}")
            elif tokenizer_content_sha256(tokenizer_config) != MISTRAL_SERVER_TOKENIZER_CONTENT_SHA256:
                errors.append("Mistral tokenizer settings differ from the validated server beyond chat_template")
            elif sha256(tokenizer_config) != MISTRAL_SERVER_TOKENIZER_CONFIG_SHA256:
                warnings.append(
                    "Mistral official tokenizer adds chat_template; training and inference "
                    "explicitly use the validated historical plain-role prompt"
                )

            recovered_config = root / "tokenizer_config.json"
            if recovered_config.is_file() and sha256(recovered_config) != MISTRAL_SERVER_TOKENIZER_CONFIG_SHA256:
                errors.append("recovered server tokenizer_config.json checksum mismatch")

    pubchem_candidates = [
        root / "full_dataset.csv",
        root / "data/pubchem/full_dataset_coarse_labels.csv",
    ]
    if args.require_data and not any(path.is_file() for path in pubchem_candidates):
        errors.append(
            "missing PubChem data: provide full_dataset.csv or "
            "data/pubchem/full_dataset_coarse_labels.csv"
        )

    nite_candidates = [
        root / "data/nite_japan_ghs/source/list_nite_all_e.xlsx",
        root / "external_validation/nite_japan_ghs/source/list_nite_all_e.xlsx",
    ]
    nite = next((path for path in nite_candidates if path.is_file()), nite_candidates[0])
    if nite.is_file():
        expected = "79e5cba0a79af3e60da39d9bbcc0e13c2581ab90b11dba90582b7e7cb720777c"
        actual = sha256(nite)
        if actual != expected:
            errors.append(f"NITE workbook checksum mismatch: {actual}")
    elif args.require_data:
        errors.append("missing NITE workbook in data/ or external_validation/")

    for warning in warnings:
        print(f"SETUP WARNING: {warning}")
    if errors:
        print("SETUP CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("SETUP CHECK PASSED")


if __name__ == "__main__":
    main()
