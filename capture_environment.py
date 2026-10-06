from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "numpy",
    "pandas",
    "scikit-learn",
    "rdkit",
    "openpyxl",
    "matplotlib",
    "chemprop",
    "datasets",
    "trl",
    "tokenizers",
    "sentencepiece",
    "protobuf",
]


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture one named runtime environment without overwriting others."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Logical runtime name, for example chem2, chem, or chemprop.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reproducibility/environments"),
    )
    args = parser.parse_args()

    output = args.output_root / args.name
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "environment_name": args.name,
        "python_executable": Path(sys.executable).name,
        "environment_prefix_name": Path(sys.prefix).name,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in PACKAGES},
    }
    try:
        report["nvidia_smi"] = subprocess.run(
            ["nvidia-smi"], check=False, capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        report["nvidia_smi"] = None
    (output / "environment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "nvidia_smi.txt").write_text(
        report.get("nvidia_smi") or "nvidia-smi unavailable\n", encoding="utf-8"
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        stdout=(output / "pip_freeze.txt").open("w", encoding="utf-8"),
    )
    print(f"Saved {args.name} environment metadata under {output.resolve()}")


if __name__ == "__main__":
    main()
