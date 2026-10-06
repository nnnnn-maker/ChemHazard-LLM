from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


LABELS = [
    "flammable",
    "oxidizing",
    "gas_under_pressure",
    "corrosive",
    "acute_toxicity",
    "irritant_harmful",
    "cmr",
    "stot",
    "environmental_hazard",
]

SYSTEM_PROMPT = (
    "You are a chemical hazard classification assistant. Given structured chemical "
    "information, predict the nine coarse-grained hazard labels. Return only one JSON "
    "object with exactly these keys: flammable, oxidizing, gas_under_pressure, corrosive, "
    "acute_toxicity, irritant_harmful, cmr, stot, environmental_hazard. Each value must be 0 or 1."
)

NITE_DOWNLOAD_PAGE = "https://www.chem-info.nite.go.jp/chem/english/ghs/ghs_nite_download_e.html"
NITE_LATEST_URL = "https://www.chem-info.nite.go.jp/chem/english/ghs/files/list_nite_all_e.xlsx"
NITE_RATIONALE_URL = "https://www.chem-info.nite.go.jp/chem/english/ghs/files/list_rational_e.xlsx"
PUBCHEM_IDEXCHANGE_URL = "https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi"
PUBCHEM_PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

UNKNOWN_TEXT = {"", "-", "classification not possible", "nan", "none"}
EXPLICIT_NEGATIVE_PREFIXES = ("not classified",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_cas(value: Any) -> str:
    text = normalize_text(value)
    match = re.search(r"\b\d{2,7}-\d{2}-\d\b", text)
    return match.group(0) if match else ""


def status_for_cell(value: Any, positive_pattern: str) -> int | None:
    text = normalize_text(value)
    lowered = text.lower()
    if lowered in UNKNOWN_TEXT:
        return None
    if re.search(positive_pattern, text, flags=re.I):
        return 1
    if lowered.startswith(EXPLICIT_NEGATIVE_PREFIXES):
        return 0
    # A classification in the same GHS class but outside the project's H-code rule is
    # a known negative for that target label (e.g., oxidizing category 1 is H271, not H272).
    if re.search(r"\bcategory\b|\btype\s+[a-g]\b|\b(?:compressed|dissolved|liquefied|refrigerated)\s+gas\b", text, flags=re.I):
        return 0
    return None


def combine_statuses(statuses: list[int | None]) -> int | None:
    if any(value == 1 for value in statuses):
        return 1
    if statuses and all(value == 0 for value in statuses):
        return 0
    return None


def compute_labels(row: pd.Series) -> tuple[dict[str, int | None], dict[str, str]]:
    specs: dict[str, list[tuple[str, str]]] = {
        "flammable": [
            ("Flammable gases", r"Category\s+1(?:A|B)?\b"),
            ("Flammable liquids", r"Category\s+(?:1|2|3|4)\b"),
            ("Flammable solids", r"Category\s+(?:1|2)\b"),
        ],
        "oxidizing": [
            ("Oxidizing liquids", r"Category\s+(?:2|3)\b"),
            ("Oxidizing solids", r"Category\s+(?:2|3)\b"),
        ],
        "gas_under_pressure": [
            ("Gases under pressure", r"(?:Compressed|Dissolved|(?:High pressure |Low pressure )?Liquefied)\s+gas\b"),
        ],
        "corrosive": [
            ("Corrosive to metals", r"Category\s+1\b"),
            ("Skin corrosion/irritation", r"Category\s+1(?:A|B|C)?\b"),
            ("Serious eye damage/eye irritation", r"Category\s+1\b"),
        ],
        "acute_toxicity": [
            ("Acute toxicity (Oral)", r"Category\s+(?:1|2|3|4|5)\b"),
            ("Acute toxicity (Dermal)", r"Category\s+(?:1|2|3|4|5)\b"),
            ("Acute toxicity (Inhalation: Gases)", r"Category\s+(?:1|2|3|4)\b"),
            ("Acute toxicity (Inhalation: Vapours)", r"Category\s+(?:1|2|3|4)\b"),
            ("Acute toxicity (Inhalation: Dusts and mists)", r"Category\s+(?:1|2|3|4)\b"),
            ("Aspiration hazard", r"Category\s+1\b"),
        ],
        "irritant_harmful": [
            ("Skin corrosion/irritation", r"Category\s+(?:2|3)\b"),
            ("Serious eye damage/eye irritation", r"Category\s+2(?:A|B)?\b"),
            ("Respiratory sensitization", r"Category\s+1(?:A|B)?\b"),
            ("Skin sensitization", r"Category\s+1(?:A|B)?\b"),
            ("Specific target organ toxicity - Single exposure", r"Category\s+3\b"),
        ],
        "cmr": [
            ("Germ cell mutagenicity", r"Category\s+(?:1|1A|1B|2)\b"),
            ("Carcinogenicity", r"Category\s+(?:1|1A|1B|2)\b"),
            ("Reproductive toxicity", r"Category\s+(?:1|1A|1B|2)\b|Effects on or via lactation"),
        ],
        "stot": [
            ("Specific target organ toxicity - Single exposure", r"Category\s+(?:1|2)\b"),
            ("Specific target organ toxicity - Repeated exposure", r"Category\s+(?:1|2)\b"),
        ],
        "environmental_hazard": [
            ("Hazardous to the aquatic environment Short term (Acute)", r"Category\s+(?:1|2|3)\b"),
            ("Hazardous to the aquatic environment Long term (Chronic)", r"Category\s+(?:1|2|3|4)\b"),
        ],
    }
    labels: dict[str, int | None] = {}
    evidence: dict[str, str] = {}
    for label, components in specs.items():
        statuses = [status_for_cell(row.get(column), pattern) for column, pattern in components]
        labels[label] = combine_statuses(statuses)
        evidence[label] = " | ".join(
            f"{column}={normalize_text(row.get(column)) or '<blank>'}" for column, _ in components
        )
    return labels, evidence


def multipart_body(fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = "----CodexNITE" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def urlopen_retry(request: urllib.request.Request | str, attempts: int = 5, timeout: int = 120) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # network retry boundary
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"Network request failed after {attempts} attempts: {last_error}")


def resolve_cas_to_cids(cas_values: list[str], cache_path: Path) -> dict[str, list[int]]:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return {key: [int(value) for value in values] for key, values in cached.items()}

    output: dict[str, list[int]] = defaultdict(list)
    # Small jobs usually finish synchronously; this avoids relying on PubChem's
    # browser-only JavaScript polling page for larger asynchronous submissions.
    for start in range(0, len(cas_values), 400):
        batch = cas_values[start : start + 400]
        fields = {
            "inputtype": "synofiltered",
            "inputdsn": "",
            "idinput": "str",
            "idstr": "\n".join(batch),
            "operatortype": "samecid",
            "outputtype": "cid",
            "outputdsn": "",
            "method": "file-pair",
            "compression": "none",
            "submitjob": "Submit Job",
        }
        body, boundary = multipart_body(fields)
        req = urllib.request.Request(
            PUBCHEM_IDEXCHANGE_URL,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": "ChemHazard-LLM-external-audit/1.0"},
        )
        response_text = urlopen_retry(req).decode("utf-8", errors="replace")
        links: list[str] = []
        for _ in range(120):
            links = re.findall(r"https://pubchem\.ncbi\.nlm\.nih\.gov/rest/download/\.fetch/[^\"'<]+\.txt", response_text)
            if links:
                break
            progress_links = re.findall(r'document\.location\.replace\("([^\"]+)"\)', response_text)
            if not progress_links:
                break
            time.sleep(2)
            progress_url = html.unescape(progress_links[0]).replace(" ", "%20")
            response_text = urlopen_retry(progress_url).decode("utf-8", errors="replace")
        if not links:
            debug_path = cache_path.with_name(f"idexchange_inprogress_{start}.html")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(response_text, encoding="utf-8")
            raise RuntimeError(f"PubChem Identifier Exchange did not return a download link: {response_text[:500]}")
        pair_text = urlopen_retry(html.unescape(links[0])).decode("utf-8", errors="replace")
        for line in pair_text.splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[1].isdigit():
                output[parts[0]].append(int(parts[1]))
        time.sleep(0.4)

    normalized = {cas: sorted(set(output.get(cas, []))) for cas in cas_values}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def fetch_pubchem_properties(cids: list[int], cache_path: Path) -> dict[int, dict[str, Any]]:
    properties: dict[int, dict[str, Any]] = {}
    if cache_path.exists():
        properties = {int(k): v for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}
    missing = [cid for cid in cids if cid not in properties]
    names = (
        "Title,IUPACName,MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,"
        "HBondAcceptorCount,RotatableBondCount,Complexity,InChI,InChIKey,ConnectivitySMILES,SMILES,HeavyAtomCount"
    )
    for start in range(0, len(missing), 100):
        batch = missing[start : start + 100]
        cid_part = ",".join(str(value) for value in batch)
        url = f"{PUBCHEM_PUG_BASE}/compound/cid/{cid_part}/property/{names}/JSON"
        payload = json.loads(urlopen_retry(url).decode("utf-8"))
        for item in payload.get("PropertyTable", {}).get("Properties", []):
            properties[int(item["CID"])] = item
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(properties, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.25)
    return properties


def aggregate_resolved_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["InChIKey"]].append(row)
    clean: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for inchikey, members in groups.items():
        merged = dict(members[0])
        merged["source_row_count"] = len(members)
        merged["nite_ids"] = "; ".join(sorted({str(m["nite_id"]) for m in members if m.get("nite_id")}))
        merged["cas_numbers"] = "; ".join(sorted({str(m["CAS"]) for m in members if m.get("CAS")}))
        merged["nite_substance_names"] = "; ".join(sorted({str(m["nite_substance_name"]) for m in members if m.get("nite_substance_name")}))
        has_conflict = False
        for label in LABELS:
            known = {m[f"label_{label}"] for m in members if m[f"label_{label}"] is not None}
            if len(known) > 1:
                has_conflict = True
            merged[f"label_{label}"] = next(iter(known)) if len(known) == 1 else None
            merged[f"valid_{label}"] = int(len(known) == 1)
            merged[f"nite_evidence_{label}"] = " || ".join(sorted({m[f"nite_evidence_{label}"] for m in members}))
        merged["num_valid_labels"] = sum(merged[f"valid_{label}"] for label in LABELS)
        merged["num_positive_labels"] = sum(int(merged[f"label_{label}"] == 1) for label in LABELS)
        if has_conflict:
            merged["resolution_status"] = "excluded_label_conflict_after_inchikey_merge"
            conflicts.append(merged)
        else:
            clean.append(merged)
    return clean, conflicts


def write_csv(path: Path, records: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        pd.DataFrame(columns=columns or []).to_csv(path, index=False, encoding="utf-8-sig")
        return
    frame = pd.DataFrame(records)
    if columns:
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def json_safe(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def make_prompt(row: dict[str, Any]) -> str:
    fields = [
        ("CID", row.get("CID")),
        ("ConnectivitySMILES", row.get("ConnectivitySMILES")),
        ("IUPACName", row.get("IUPACName")),
        ("MolecularFormula", row.get("MolecularFormula")),
        ("MolecularWeight", row.get("MolecularWeight")),
        ("XLogP", row.get("XLogP")),
        ("TPSA", row.get("TPSA")),
    ]
    lines = ["Classify the chemical into the predefined coarse-grained hazard categories.", "Chemical information:"]
    lines.extend(f"- {key}: {value}" for key, value in fields if value not in (None, "") and not pd.isna(value))
    lines.append("Return only JSON.")
    return "\n".join(lines)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            labels = {label: json_safe(row.get(f"label_{label}")) for label in LABELS}
            mask = {label: int(row.get(f"valid_{label}", 0)) for label in LABELS}
            record = {
                "id": f"NITE:{row.get('nite_ids') or row.get('nite_id')}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": make_prompt(row)},
                ],
                "labels": labels,
                "label_mask": mask,
                "provenance": {
                    "dataset": "NITE Japan-GHS latest classifications",
                    "nite_ids": row.get("nite_ids"),
                    "cas_numbers": row.get("cas_numbers"),
                    "pubchem_cid": json_safe(row.get("CID")),
                    "inchikey": row.get("InChIKey"),
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a source-audited, InChIKey-decontaminated NITE Japan-GHS external test set.")
    parser.add_argument("--nite-xlsx", type=Path, required=True)
    parser.add_argument("--nite-rationale-xlsx", type=Path)
    parser.add_argument("--project-reference-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.output_dir / "cache"
    nite = pd.read_excel(args.nite_xlsx, sheet_name="Latest GHS clasification")
    nite.insert(0, "nite_source_row", range(2, len(nite) + 2))
    nite["CAS_normalized"] = nite["CAS"].map(normalize_cas)

    cas_values = sorted({value for value in nite["CAS_normalized"] if value})
    cas_to_cids = resolve_cas_to_cids(cas_values, cache_dir / "pubchem_cas_to_cids.json")
    all_cids = sorted({cid for values in cas_to_cids.values() for cid in values})
    cid_properties = fetch_pubchem_properties(all_cids, cache_dir / "pubchem_cid_properties.json")

    resolved_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    for _, source in nite.iterrows():
        labels, evidence = compute_labels(source)
        base: dict[str, Any] = {
            "nite_source_row": int(source["nite_source_row"]),
            "nite_id": normalize_text(source.get("ID")),
            "CAS": source["CAS_normalized"],
            "nite_substance_name": normalize_text(source.get("Substance Name")),
            "nite_detail_url": normalize_text(source.get("Detail (NITE HP URL)")),
        }
        for label in LABELS:
            base[f"label_{label}"] = labels[label]
            base[f"valid_{label}"] = int(labels[label] is not None)
            base[f"nite_evidence_{label}"] = evidence[label]
        cas = base["CAS"]
        if not cas:
            unresolved_rows.append({**base, "resolution_status": "excluded_missing_cas"})
            continue
        candidates = [cid_properties[cid] for cid in cas_to_cids.get(cas, []) if cid in cid_properties and cid_properties[cid].get("InChIKey")]
        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in candidates:
            by_key[str(item["InChIKey"]).upper()].append(item)
        if not by_key:
            unresolved_rows.append({**base, "pubchem_candidate_cids": ";".join(map(str, cas_to_cids.get(cas, []))), "resolution_status": "excluded_pubchem_unresolved"})
            continue
        if len(by_key) > 1:
            unresolved_rows.append({
                **base,
                "pubchem_candidate_cids": ";".join(str(item["CID"]) for item in candidates),
                "pubchem_candidate_inchikeys": ";".join(sorted(by_key)),
                "resolution_status": "excluded_ambiguous_multiple_inchikeys",
            })
            continue
        inchikey, same_structure = next(iter(by_key.items()))
        item = sorted(same_structure, key=lambda value: int(value["CID"]))[0]
        record = {**base, **item}
        record["InChIKey"] = inchikey
        record["resolution_status"] = "resolved_unique_inchikey"
        record["pubchem_candidate_cids"] = ";".join(str(value["CID"]) for value in same_structure)
        resolved_rows.append(record)

    aggregated, conflicts = aggregate_resolved_rows(resolved_rows)
    unresolved_rows.extend(conflicts)

    reference = pd.read_csv(args.project_reference_csv, low_memory=False)
    if "num_coarse_labels" in reference.columns:
        reference = reference[pd.to_numeric(reference["num_coarse_labels"], errors="coerce").fillna(0) > 0]
    reference_keys = {normalize_text(value).upper() for value in reference["InChIKey"] if normalize_text(value)}
    reference_blocks = {value[:14] for value in reference_keys if len(value) >= 14}

    audit_ready: list[dict[str, Any]] = []
    for row in aggregated:
        key = row["InChIKey"].upper()
        row["inchikey_overlap_seed123"] = int(key in reference_keys)
        row["connectivity_block_overlap_seed123"] = int(key[:14] in reference_blocks)
        row["decontamination_status"] = (
            "excluded_exact_inchikey_overlap" if key in reference_keys else
            "retained_exact_disjoint_connectivity_overlap" if key[:14] in reference_blocks else
            "retained_connectivity_disjoint"
        )
        audit_ready.append(row)

    exact_disjoint = [row for row in audit_ready if not row["inchikey_overlap_seed123"] and row["num_valid_labels"] > 0]
    connectivity_disjoint = [row for row in exact_disjoint if not row["connectivity_block_overlap_seed123"]]
    excluded_overlap = [row for row in audit_ready if row["inchikey_overlap_seed123"]]

    identity_columns = [
        "nite_ids", "cas_numbers", "nite_substance_names", "nite_detail_url", "CID", "Title", "IUPACName",
        "MolecularFormula", "MolecularWeight", "XLogP", "TPSA", "ConnectivitySMILES", "SMILES", "InChI", "InChIKey",
        "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount", "HeavyAtomCount", "Complexity",
        "source_row_count", "pubchem_candidate_cids", "num_valid_labels", "num_positive_labels",
        "inchikey_overlap_seed123", "connectivity_block_overlap_seed123", "decontamination_status",
    ]
    label_columns = [f"label_{label}" for label in LABELS]
    mask_columns = [f"valid_{label}" for label in LABELS]
    evidence_columns = [f"nite_evidence_{label}" for label in LABELS]
    output_columns = identity_columns + label_columns + mask_columns + evidence_columns

    write_csv(args.output_dir / "nite_external_test_exact_inchikey_disjoint.csv", exact_disjoint, output_columns)
    write_csv(args.output_dir / "nite_external_test_connectivity_disjoint.csv", connectivity_disjoint, output_columns)
    write_csv(args.output_dir / "nite_external_resolved_audit.csv", audit_ready, output_columns)
    write_csv(args.output_dir / "nite_external_excluded_overlap.csv", excluded_overlap, output_columns)
    write_csv(args.output_dir / "nite_external_unresolved_or_conflict.csv", unresolved_rows)
    write_jsonl(args.output_dir / "nite_external_test_exact_inchikey_disjoint.jsonl", exact_disjoint)
    write_jsonl(args.output_dir / "nite_external_test_connectivity_disjoint.jsonl", connectivity_disjoint)

    label_summary: dict[str, dict[str, int]] = {}
    for label in LABELS:
        label_summary[label] = {
            "valid": sum(int(row[f"valid_{label}"]) for row in exact_disjoint),
            "positive": sum(int(row[f"label_{label}"] == 1) for row in exact_disjoint),
            "negative": sum(int(row[f"label_{label}"] == 0) for row in exact_disjoint),
        }

    summary = {
        "dataset_name": "NITE Japan-GHS external test set for ChemHazard-LLM",
        "build_timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nite_source_rows": int(len(nite)),
        "nite_rows_with_cas": int((nite["CAS_normalized"] != "").sum()),
        "nite_unique_cas": len(cas_values),
        "pubchem_resolved_source_rows": len(resolved_rows),
        "resolved_unique_inchikeys_before_conflict_filter": len({row["InChIKey"] for row in resolved_rows}),
        "label_conflict_inchikey_groups": len(conflicts),
        "unresolved_or_conflict_rows": len(unresolved_rows),
        "seed123_reference_rows": int(len(reference)),
        "seed123_reference_unique_inchikeys": len(reference_keys),
        "exact_inchikey_overlap_excluded": len(excluded_overlap),
        "exact_inchikey_disjoint_rows": len(exact_disjoint),
        "connectivity_block_disjoint_rows": len(connectivity_disjoint),
        "exact_disjoint_rows_with_at_least_one_positive": sum(row["num_positive_labels"] > 0 for row in exact_disjoint),
        "exact_disjoint_complete_case_rows": sum(row["num_valid_labels"] == len(LABELS) for row in exact_disjoint),
        "label_summary_exact_disjoint": label_summary,
        "notes": [
            "Labels are H-code-compatible coarse mappings, not a broad union of all NITE hazard categories.",
            "Classification not possible and '-' are masked as unknown, never converted to negative.",
            "The primary set removes full InChIKey overlap with all 11,556 seed123 experiment compounds.",
            "The connectivity-disjoint subset additionally removes first-block InChIKey overlap.",
            "Because the historical PubChem collection did not preserve per-label contributing-source attribution, source independence is audited at the NITE-file level plus exact-structure decontamination; complete upstream provenance independence cannot be proven retrospectively.",
        ],
    }
    (args.output_dir / "nite_external_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    source_audit = {
        "nite_download_page": NITE_DOWNLOAD_PAGE,
        "nite_latest_classification_url": NITE_LATEST_URL,
        "nite_latest_page_update_date": "2026-09-02",
        "nite_local_file": str(args.nite_xlsx.resolve()),
        "nite_local_sha256": sha256_file(args.nite_xlsx),
        "nite_rationale_url": NITE_RATIONALE_URL,
        "nite_rationale_local_file": str(args.nite_rationale_xlsx.resolve()) if args.nite_rationale_xlsx and args.nite_rationale_xlsx.exists() else None,
        "nite_rationale_sha256": sha256_file(args.nite_rationale_xlsx) if args.nite_rationale_xlsx and args.nite_rationale_xlsx.exists() else None,
        "pubchem_identifier_resolution": {
            "identifier_exchange_url": PUBCHEM_IDEXCHANGE_URL,
            "input_type": "synofiltered (CAS as synonym)",
            "output_type": "CID, two-column correspondence",
            "property_api": f"{PUBCHEM_PUG_BASE}/compound/cid/<batch>/property/<properties>/JSON",
            "ambiguity_policy": "retain only CAS mappings resolving to one unique full InChIKey; multiple CIDs sharing that InChIKey are collapsed",
        },
        "project_reference_file": str(args.project_reference_csv.resolve()),
        "project_reference_sha256": sha256_file(args.project_reference_csv),
        "project_reference_filter": "num_coarse_labels > 0 (seed123 experiment universe, 11,556 rows)",
        "decontamination": {
            "primary": "exclude full InChIKey matches",
            "strict_subset": "also exclude first 14-character InChIKey connectivity-block matches",
        },
        "label_mapping_version": "chemhazard_hcode_compatible_v1",
        "generated_files": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
    }
    (args.output_dir / "nite_external_source_audit.json").write_text(json.dumps(source_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
