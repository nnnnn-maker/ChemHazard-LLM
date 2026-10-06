from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IDENTITY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_revision(path: Path) -> str | None:
    for part in path.resolve().parts:
        if re.fullmatch(r"[0-9a-f]{40}", part):
            return part
    if (path / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value):
            return value
    return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture local model identities and checksums for publication."
    )
    parser.add_argument("--spec", type=Path, default=Path("MODEL_MANIFEST.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--model-dir",
        action="append",
        default=[],
        metavar="CLI_NAME=PATH",
        help="Override one model directory after verifying the experiment-used path.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reproducibility/model_manifest.local.json")
    )
    parser.add_argument(
        "--public-output",
        type=Path,
        default=None,
        help="Also write a sanitized checksum manifest without local filesystem paths.",
    )
    parser.add_argument(
        "--hash-weights",
        action="store_true",
        help="Also hash all .safetensors and .bin files. This can take several minutes.",
    )
    args = parser.parse_args()

    overrides: dict[str, Path] = {}
    for entry in args.model_dir:
        name, separator, value = entry.partition("=")
        if not separator or not name or not value:
            parser.error("--model-dir must have the form CLI_NAME=PATH")
        overrides[name] = Path(value)

    specification = read_json(args.spec)
    known_names = {model["cli_name"] for model in specification["models"]}
    unknown_names = sorted(set(overrides) - known_names)
    if unknown_names:
        parser.error(f"unknown --model-dir names: {', '.join(unknown_names)}")
    captured: list[dict[str, Any]] = []
    missing: list[str] = []
    unresolved_revisions: list[str] = []
    unverified_snapshots: list[str] = []

    for model in specification["models"]:
        directory = overrides.get(model["cli_name"], args.root / model["local_directory"])
        item: dict[str, Any] = {**model, "resolved_directory": str(directory)}
        if not directory.is_dir():
            item["status"] = "missing"
            missing.append(model["cli_name"])
            captured.append(item)
            continue

        config_path = directory / "config.json"
        if not config_path.is_file():
            item["status"] = "missing_config"
            missing.append(model["cli_name"])
            captured.append(item)
            continue

        config = read_json(config_path)
        architecture = model["architecture"]
        architectures = config.get("architectures") or []
        identity_ok = (
            config.get("model_type") == model["model_type"]
            and architecture in architectures
        )
        checksums = {
            name: sha256(directory / name)
            for name in IDENTITY_FILES
            if (directory / name).is_file()
        }
        if model.get("model_snapshot_verified") is False and not model.get("experiment_prompt_equivalent_verified"):
            unverified_snapshots.append(model["cli_name"])
        expected_tokenizer_hash = model.get("server_tokenizer_config_sha256")
        revision = model.get("used_revision") or infer_revision(directory)
        revision_valid = bool(
            isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)
        )
        if not revision_valid:
            unresolved_revisions.append(model["cli_name"])

        weight_files = sorted(
            path
            for suffix in ("*.safetensors", "*.bin")
            for path in directory.glob(suffix)
        )
        weight_sha256 = (
            {path.name: sha256(path) for path in weight_files}
            if args.hash_weights else None
        )
        expected_weights = model.get("experiment_safetensors_sha256", {})
        weight_fingerprint_match = (
            all(weight_sha256.get(name) == expected for name, expected in expected_weights.items())
            if weight_sha256 is not None and expected_weights else None
        )
        tokenizer_match = (
            checksums.get("tokenizer_config.json") == expected_tokenizer_hash
            if expected_tokenizer_hash else None
        )
        if identity_ok and (tokenizer_match is False or weight_fingerprint_match is False):
            item["status"] = "experiment_fingerprint_mismatch"
        item.update(
            {
                "status": item.get("status", "ok" if identity_ok else "identity_mismatch"),
                "used_revision": revision,
                "revision_valid": revision_valid,
                "observed_model_type": config.get("model_type"),
                "observed_architectures": architectures,
                "identity_file_sha256": checksums,
                "server_tokenizer_config_match": tokenizer_match,
                "experiment_weight_fingerprint_match": weight_fingerprint_match,
                "weight_file_count": len(weight_files),
                "weight_bytes": sum(path.stat().st_size for path in weight_files),
                "weight_sha256": weight_sha256,
            }
        )
        captured.append(item)

    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "specification": str(args.spec),
        "models": captured,
        "missing_models": missing,
        "unresolved_revisions": unresolved_revisions,
        "unverified_snapshots": unverified_snapshots,
        "publication_ready": not missing
        and not unresolved_revisions
        and not unverified_snapshots
        and all(item.get("status") == "ok" for item in captured),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")
    if args.public_output is not None:
        if missing or any(
            item.get("status") != "ok" or not item.get("weight_sha256")
            for item in captured
        ):
            raise SystemExit(
                "A public manifest requires all five verified models and --hash-weights"
            )
        public_models = []
        for item in captured:
            public_models.append(
                {
                    "cli_name": item["cli_name"],
                    "display_name": item["display_name"],
                    "repository_id": item.get("repository_id_used", item.get("repository_id")),
                    "used_revision": item["used_revision"],
                    "experiment_access_date": item.get("experiment_access_date"),
                    "experiment_access_date_evidence": item.get("experiment_access_date_evidence"),
                    "revision_evidence": item.get("revision_evidence"),
                    "model_snapshot_verified": item.get("model_snapshot_verified"),
                    "experiment_prompt_equivalent_verified": item.get("experiment_prompt_equivalent_verified"),
                    "observed_model_type": item.get("observed_model_type"),
                    "observed_architectures": item.get("observed_architectures"),
                    "identity_file_sha256": item.get("identity_file_sha256"),
                    "weight_file_count": item.get("weight_file_count"),
                    "weight_sha256": item["weight_sha256"],
                }
            )
        public_payload = {
            "schema_version": 1,
            "source_capture_utc": payload["captured_at_utc"],
            "notes": (
                "Sanitized from the validated server capture. The Mistral weight hashes "
                "match the pinned Instruct revision. The recovered experiment tokenizer "
                "configuration differs only by lacking chat_template; historical plain-role "
                "prompting is explicit in the reproduction code. Full directories are not byte-identical."
                if any(item.get("experiment_prompt_equivalent_verified") for item in captured)
                else "Sanitized from the validated server capture; local filesystem paths removed."
            ),
            "models": public_models,
        }
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(
            json.dumps(public_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved: {args.public_output}")
    if not payload["publication_ready"]:
        print(
            "Manifest is not publication-ready. Resolve missing or mismatched "
            "models and any unverified model snapshots."
        )


if __name__ == "__main__":
    main()
