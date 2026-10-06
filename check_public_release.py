from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".pyc"}
FORBIDDEN_PARTS = {
    "__pycache__",
    "Model",
    "checkpoints_lora_sft",
    "checkpoints_chemprop",
    "predictions",
    "evaluation_llm_coarse",
}
RESTRICTED_PUBLIC_PATHS = {
    "tokenizer_config.json",
    "data/nite_japan_ghs/source/list_nite_all_e.xlsx",
    "data/pubchem/full_dataset_coarse_labels.csv",
    "data/pubchem/seed123_split_manifest.csv",
}
RESTRICTED_PUBLIC_GLOBS = (
    "data/pubchem/seed123_simple/*.jsonl",
    "data/pubchem/seed123_pictograms_rare2x/*.jsonl",
    "data/nite_japan_ghs/*.csv",
    "data/nite_japan_ghs/*.jsonl",
    "data/nite_japan_ghs/*.xlsx",
    "supplementary/Additional_file_*.docx",
    "supplementary/Additional_file_*.xlsx",
)
LOCAL_PATH_PATTERNS = (
    re.compile(r"C:\\Users\\", re.IGNORECASE),
    re.compile(r"/home/featurize/"),
    re.compile(r"/Users/[^/]+/"),
)
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
)
PLACEHOLDER_PATTERN = re.compile(r"REPLACE_|YYYY-MM-DD")


def iter_files(root: Path):
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            for name in result.stdout.decode("utf-8").split("\0"):
                if name:
                    path = root / name
                    if path.is_file():
                        yield path
            return
    for path in root.rglob("*"):
        if (
            path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
        ):
            yield path


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_restricted_public_path(rel: str) -> bool:
    return (
        rel in RESTRICTED_PUBLIC_PATHS
        or rel == "full_dataset.csv"
        or rel.startswith("processed_coarse/")
        or rel.startswith("sft_coarse_jsonl_")
        or rel.startswith("sft_clean_ablation_")
        or rel.startswith("data/nite_japan_ghs/source/")
        or rel.endswith((".jsonl", ".xlsx", ".docx"))
        or (rel.endswith(".csv") and rel != "supplementary/scaffold_paired_bootstrap_delta.csv")
        or any(fnmatch.fnmatchcase(rel, pattern) for pattern in RESTRICTED_PUBLIC_GLOBS)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a public GitHub/Zenodo release.")
    parser.add_argument(
        "--final",
        action="store_true",
        help="Require final license, citation, Zenodo metadata, revisions, and version.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    errors: list[str] = []
    warnings: list[str] = []

    required = {
        "README.md",
        ".gitignore",
        ".gitattributes",
        "VERSION",
        "CHANGELOG.md",
        "MODEL_MANIFEST.json",
        "reproducibility/model_manifest.public.json",
        "RUNTIME_ENVIRONMENTS.json",
        "DATASET_CARD.md",
        "GITHUB_ZENODO_RELEASE.md",
        "PUBLIC_RELEASE_CONTENTS.md",
    }
    for name in sorted(required):
        if not (root / name).is_file():
            errors.append(f"missing required release file: {name}")

    git_index_exists = (root / ".git").exists()
    if git_index_exists:
        unstaged = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", "--"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if unstaged.returncode != 0:
            errors.append("could not compare tracked working files with the Git index")
        else:
            for name in unstaged.stdout.splitlines():
                errors.append(f"tracked file has unstaged changes; inspect and stage it: {name}")
    for path in iter_files(root):
        rel = relative(path, root)
        if git_index_exists and is_restricted_public_path(rel):
            errors.append(f"restricted file is tracked for the code-only release: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden generated/model file: {rel}")
        if any(part in FORBIDDEN_PARTS for part in path.relative_to(root).parts):
            errors.append(f"forbidden directory content: {rel}")
        if path.stat().st_size > 100 * 1024 * 1024:
            errors.append(f"file exceeds GitHub 100 MiB limit: {rel}")
        elif path.stat().st_size > 25 * 1024 * 1024:
            warnings.append(f"file exceeds browser-upload limit; use Git CLI: {rel}")

        if path.name == Path(__file__).name:
            continue
        if path.suffix.lower() not in {".xlsx", ".png", ".jpg", ".jpeg", ".pdf"}:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for pattern in LOCAL_PATH_PATTERNS:
                if pattern.search(text):
                    errors.append(f"local absolute path found in: {rel}")
                    break
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    errors.append(f"possible credential found in: {rel}")
                    break

    for rel in sorted(RESTRICTED_PUBLIC_PATHS):
        if (root / rel).is_file():
            warnings.append(
                f"locally present but must remain ignored until redistribution is cleared: {rel}"
            )

    manifest_path = root / "MODEL_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        public_path = root / "reproducibility/model_manifest.public.json"
        public_models = {}
        if public_path.is_file():
            public_manifest = json.loads(public_path.read_text(encoding="utf-8"))
            public_models = {
                model["cli_name"]: model for model in public_manifest.get("models", [])
            }
        for model in manifest.get("models", []):
            name = model.get("cli_name", "unknown")
            published = public_models.get(name)
            if published is None:
                errors.append(f"model {name} missing from sanitized public model manifest")
            elif (
                published.get("used_revision") != model.get("used_revision")
                or published.get("experiment_access_date") != model.get("experiment_access_date")
                or published.get("experiment_access_date_evidence")
                != model.get("experiment_access_date_evidence")
                or published.get("display_name") != model.get("display_name")
                or published.get("model_snapshot_verified")
                != model.get("model_snapshot_verified")
                or published.get("experiment_prompt_equivalent_verified")
                != model.get("experiment_prompt_equivalent_verified")
                or not published.get("weight_sha256")
            ):
                errors.append(f"model {name} public identity or weight hashes are incomplete or inconsistent")
            elif model.get("server_tokenizer_config_sha256") and (
                published.get("identity_file_sha256", {}).get("tokenizer_config.json")
                != model["server_tokenizer_config_sha256"]
            ):
                errors.append(f"model {name} public tokenizer hash does not match the server capture")
            elif any(
                published.get("weight_sha256", {}).get(filename) != expected
                for filename, expected in model.get("experiment_safetensors_sha256", {}).items()
            ):
                errors.append(f"model {name} public weight hashes do not match the experiment")
            revision = model.get("used_revision")
            access_date = model.get("experiment_access_date")
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                (errors if args.final else warnings).append(
                    f"model {name} lacks a validated 40-character used_revision"
                )
            if not isinstance(access_date, str) or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}", access_date
            ):
                (errors if args.final else warnings).append(
                    f"model {name} lacks experiment_access_date in YYYY-MM-DD format"
                )
            evidence = model.get("experiment_access_date_evidence") or {}
            try:
                observed_utc = datetime.fromisoformat(evidence["timestamp_utc"].replace("Z", "+00:00"))
                observed_china_date = observed_utc.astimezone(
                    timezone(timedelta(hours=8))
                ).date().isoformat()
                if (
                    evidence.get("source") != "huggingface_local_download_metadata"
                    or observed_china_date != access_date
                    or not evidence.get("metadata_file", "").endswith(".metadata")
                    or evidence.get("matching_revision_records", 0) < 1
                ):
                    raise ValueError("metadata evidence is incomplete or date is inconsistent")
            except (KeyError, TypeError, ValueError):
                (errors if args.final else warnings).append(
                    f"model {name} lacks consistent source evidence for its recorded access date"
                )
            if model.get("model_snapshot_verified") is False and not model.get("experiment_prompt_equivalent_verified"):
                (errors if args.final else warnings).append(
                    f"model {name} is neither snapshot-identical nor verified as experiment-prompt-equivalent"
                )

    runtime_path = root / "RUNTIME_ENVIRONMENTS.json"
    if runtime_path.is_file():
        runtime_spec = json.loads(runtime_path.read_text(encoding="utf-8"))
        for name, details in runtime_spec.get("environments", {}).items():
            capture_dir = root / details["capture_directory"]
            environment_file = capture_dir / "environment.json"
            freeze_file = capture_dir / "pip_freeze.txt"
            nvidia_file = capture_dir / "nvidia_smi.txt"
            for path in (environment_file, freeze_file, nvidia_file):
                if not path.is_file():
                    (errors if args.final else warnings).append(
                        f"missing {name} runtime capture: {relative(path, root)}"
                    )
            if environment_file.is_file():
                captured = json.loads(environment_file.read_text(encoding="utf-8"))
                if captured.get("environment_name") != name:
                    errors.append(f"runtime capture name mismatch for {name}")
                packages = captured.get("packages", {})
                for package in details.get("required_packages", []):
                    if not packages.get(package):
                        (errors if args.final else warnings).append(
                            f"{name} runtime does not report package: {package}"
                        )

    legacy_runtime_files = (
        "reproducibility/environment.json",
        "reproducibility/pip_freeze.txt",
        "reproducibility/nvidia_smi.txt",
    )
    if any((root / name).is_file() for name in legacy_runtime_files):
        (errors if args.final else warnings).append(
            "legacy single-environment capture is present; replace it with the "
            "named chem2 and chem capture directories before release"
        )

    version = (root / "VERSION").read_text(encoding="utf-8").strip() if (root / "VERSION").is_file() else ""
    if args.final and (not re.fullmatch(r"\d+\.\d+\.\d+", version) or version.endswith("-dev")):
        errors.append("VERSION is not a final semantic version")

    final_files = {"LICENSE", "CITATION.cff", ".zenodo.json"}
    for name in sorted(final_files):
        path = root / name
        if not path.is_file():
            (errors if args.final else warnings).append(f"final metadata not yet present: {name}")
        elif PLACEHOLDER_PATTERN.search(path.read_text(encoding="utf-8")):
            errors.append(f"release placeholder remains in: {name}")

    if args.final:
        citation_path = root / "CITATION.cff"
        if citation_path.is_file():
            citation_text = citation_path.read_text(encoding="utf-8")
            cited_version = re.search(r"(?m)^version:\s*['\"]?([^\s'\"]+)", citation_text)
            if cited_version is None or cited_version.group(1) != version:
                errors.append("CITATION.cff version does not match VERSION")
            released = re.search(r"(?m)^date-released:\s*['\"]?(\d{4}-\d{2}-\d{2})", citation_text)
            try:
                if released is None:
                    raise ValueError("missing release date")
                date.fromisoformat(released.group(1))
            except ValueError:
                errors.append("CITATION.cff lacks a valid actual release date")
        zenodo_path = root / ".zenodo.json"
        if zenodo_path.is_file():
            zenodo = json.loads(zenodo_path.read_text(encoding="utf-8"))
            if zenodo.get("version") != version:
                errors.append(".zenodo.json version does not match VERSION")

    print("PUBLIC RELEASE AUDIT")
    print(f"mode: {'final' if args.final else 'development'}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")
    for item in errors:
        print(f"ERROR: {item}")
    for item in warnings:
        print(f"WARNING: {item}")
    if errors:
        raise SystemExit(1)
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
