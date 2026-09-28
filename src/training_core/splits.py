"""Report overlap for every use; enforce independence only where claimed."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import combinations
import re
from typing import Any

from training_core.contracts import DatasetSpec, PURPOSES, Sample
from training_core.data import _validate_samples, stable_hash

KEY_NAMES = ("id", "exact", "template", "group")


def _collect_keys(
    samples: list[Sample], fingerprints: Callable[[Sample], dict[str, str]], context: str,
) -> tuple[dict[str, set[str]], dict[str, int]]:
    _validate_samples(samples, context)
    keys: dict[str, set[str]] = {key: set() for key in KEY_NAMES}
    coverage = {key: 0 for key in KEY_NAMES}
    for sample in samples:
        values = fingerprints(sample)
        if not isinstance(values, Mapping):
            raise ValueError(f"Fingerprint provider must return a mapping for {context}")
        row_keys = {
            "id": sample.sample_id,
            "group": sample.group_id if sample.group_id is not None else values.get("group"),
            "exact": values.get("exact"), "template": values.get("template"),
        }
        for key, value in row_keys.items():
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                raise ValueError(f"{key} fingerprint must be a string for {context}")
            if value.strip():
                keys[key].add(value)
                coverage[key] += 1
    return keys, coverage


def _hashed_keys(keys: dict[str, set[str]]) -> dict[str, list[str]]:
    return {key: sorted(stable_hash({"kind": key, "value": value}) for value in keys[key]) for key in KEY_NAMES}


def inspect_splits(
    datasets: dict[str, list[Sample]],
    specs: dict[str, DatasetSpec],
    fingerprints: Callable[[Sample], dict[str, str]],
) -> dict[str, Any]:
    """Return deterministic unique-key overlap counts, without emitting content.

    Fingerprints supply optional ``exact`` and ``template`` keys. IDs and groups
    come from Sample. Empty keys never make unrelated records overlap.
    """
    if set(datasets) != set(specs):
        raise ValueError("Dataset names must exactly match their specifications")
    keys: dict[str, dict[str, set[str]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, Any]] = []
    independent_names: set[str] = set()
    training_present = any(specs[name].purpose == "train" and bool(rows) for name, rows in datasets.items())
    for name in sorted(datasets):
        samples, spec = datasets[name], specs[name]
        if spec.purpose not in PURPOSES:
            raise ValueError(f"Unknown dataset purpose for {name}: {spec.purpose}")
        keys[name], coverage = _collect_keys(samples, fingerprints, name)
        summaries[name] = {
            "purpose": spec.purpose, "count": len(samples),
            "independent": spec.independent, "provenance": spec.provenance,
            "fingerprint_coverage": coverage,
        }
        if spec.purpose == "independent_test":
            independent_names.add(name)
            if not spec.independent or not spec.provenance.strip() or not samples:
                violations.append({
                    "scope": "independent_test", "code": "missing_independence_evidence",
                    "datasets": [name],
                    "reason": "Independent test requires nonempty data, an attestation and provenance",
                })
    if independent_names:
        if not training_present:
            violations.append({
                "scope": "independent_test", "code": "missing_training_reference",
                "datasets": sorted(independent_names),
            })
        for name, summary in summaries.items():
            missing = [key for key in ("exact", "template") if summary["fingerprint_coverage"][key] != summary["count"]]
            if missing:
                violations.append({
                    "scope": "independent_test", "code": "incomplete_fingerprint_coverage",
                    "datasets": [name], "keys": missing,
                })
    pairs: list[dict[str, Any]] = []
    learning = {"train", "development"}
    nonindependent = learning | {"historical_regression"}
    for left, right in combinations(sorted(datasets), 2):
        left_purpose, right_purpose = specs[left].purpose, specs[right].purpose
        overlaps = {key: len(keys[left][key] & keys[right][key]) for key in ("id", "exact", "template", "group")}
        pairs.append({"left": left, "right": right, "left_purpose": left_purpose,
                      "right_purpose": right_purpose, "overlaps": overlaps})
        if not any(overlaps.values()):
            continue
        scope = None
        if left_purpose in learning and right_purpose in learning:
            scope = "training_selection"
        elif (left_purpose == "independent_test" and right_purpose in nonindependent) or (
            right_purpose == "independent_test" and left_purpose in nonindependent
        ):
            scope = "independent_test"
        if scope:
            violations.append({"scope": scope, "code": "partition_overlap", "datasets": [left, right],
                               "overlaps": overlaps})
    training_valid = not any(v["scope"] == "training_selection" for v in violations)
    independent_eligible = bool(independent_names) and not any(v["scope"] == "independent_test" for v in violations)
    return {
        "datasets": summaries, "pairs": pairs, "violations": violations,
        "training_selection_valid": training_valid,
        "training_reference_present": training_present,
        "independent_test_eligible": independent_eligible,
        "valid": not violations,
        "count_definition": "Number of distinct shared keys; missing/blank fingerprints are excluded",
    }


def validate_splits(
    report: dict[str, Any], *, for_training: bool = False, require_independent: bool = False,
) -> None:
    """Validate the requested use without blocking training on an unused test."""
    if for_training and not report["training_selection_valid"]:
        raise ValueError("Training/development partitions overlap")
    if require_independent and not report["independent_test_eligible"]:
        raise ValueError("No eligible independent test: missing evidence or partition overlap")
    if not for_training and not require_independent and not report["valid"]:
        raise ValueError("Dataset purpose/isolation checks failed")


def build_lineage(
    datasets: dict[str, list[Sample]],
    specs: dict[str, DatasetSpec],
    fingerprints: Callable[[Sample], dict[str, str]],
) -> dict[str, Any]:
    """Persist only hashed membership keys, without source paths or sample data.

    Entries describe data actually consumed by the artifact's training/selection
    process. Callers must supply every such partition, including development.
    """
    if set(datasets) != set(specs):
        raise ValueError("Dataset names must exactly match their specifications")
    entries = {}
    for name in sorted(datasets):
        if specs[name].purpose not in PURPOSES:
            raise ValueError(f"Unknown dataset purpose for {name}: {specs[name].purpose}")
        keys, coverage = _collect_keys(datasets[name], fingerprints, name)
        entries[name] = {
            "purpose": specs[name].purpose, "count": len(datasets[name]),
            "keys": _hashed_keys(keys), "coverage": coverage,
        }
    return {"version": 1, "entries": entries}


def validate_lineage(lineage: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Validate v1 membership and return normalized entries with coverage.

    Older v1 documents did not count per-row fingerprint coverage. A distinct
    key count equal to sample count proves full coverage; anything less is
    unknown, because repeated content and missing keys cannot be distinguished.
    """
    if lineage is None:
        return {}
    if (not isinstance(lineage, dict) or type(lineage.get("version")) is not int
            or lineage["version"] != 1 or not isinstance(lineage.get("entries"), dict)):
        raise ValueError("Invalid or unsupported lineage document")
    normalized = {}
    for name, entry in lineage["entries"].items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("Invalid lineage entry")
        count = entry.get("count")
        if entry.get("purpose") not in PURPOSES or not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"Invalid lineage purpose/count for {name}")
        keys = entry.get("keys")
        if not isinstance(keys, dict) or set(keys) != set(KEY_NAMES):
            raise ValueError(f"Invalid lineage keys for {name}")
        for key, values in keys.items():
            if not isinstance(values, list) or any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value) for value in values):
                raise ValueError(f"Invalid hashed lineage membership for {name}/{key}")
            if len(values) != len(set(values)) or len(values) > count:
                raise ValueError(f"Invalid lineage membership count for {name}/{key}")
        if len(keys["id"]) != count:
            raise ValueError(f"Lineage IDs/count disagree for {name}")
        if "coverage" not in entry:
            coverage = {key: count if len(keys[key]) == count else None for key in KEY_NAMES}
        else:
            coverage = entry["coverage"]
            if not isinstance(coverage, dict) or set(coverage) != set(KEY_NAMES):
                raise ValueError(f"Invalid lineage coverage for {name}")
            for key, covered in coverage.items():
                if covered is not None and (
                    not isinstance(covered, int) or isinstance(covered, bool)
                    or covered < len(keys[key]) or covered > count
                    or (covered > 0 and not keys[key])
                ):
                    raise ValueError(f"Invalid lineage coverage count for {name}/{key}")
            if coverage["id"] != count:
                raise ValueError(f"Lineage ID coverage/count disagree for {name}")
        normalized[name] = {"purpose": entry["purpose"], "count": count,
                            "keys": {key: list(values) for key, values in keys.items()},
                            "coverage": dict(coverage)}
    return normalized


def check_lineage(
    samples: list[Sample],
    spec: DatasetSpec,
    fingerprints: Callable[[Sample], dict[str, str]],
    lineage: dict[str, Any] | None,
) -> dict[str, Any]:
    """Check new evaluation data against portable, hashed artifact lineage.

    Historical regression remains runnable without lineage, but the report
    explicitly marks its absence. It never qualifies as an independent test.
    """
    if spec.purpose not in PURPOSES:
        raise ValueError(f"Unknown dataset purpose: {spec.purpose}")
    stored = validate_lineage(lineage)
    incoming_keys, coverage = _collect_keys(samples, fingerprints, "evaluation")
    incoming = {key: set(values) for key, values in _hashed_keys(incoming_keys).items()}
    all_matches: dict[str, set[str]] = {key: set() for key in KEY_NAMES}
    entries = []
    for name in sorted(stored):
        entry = stored[name]
        matches = {key: incoming[key] & set(entry["keys"][key]) for key in KEY_NAMES}
        for key in KEY_NAMES:
            all_matches[key].update(matches[key])
        entries.append({"name": name, "purpose": entry["purpose"], "count": entry["count"],
                        "fingerprint_coverage": entry["coverage"],
                        "overlaps": {key: len(matches[key]) for key in KEY_NAMES}})
    training_present = any(entry["purpose"] == "train" and entry["count"] > 0 for entry in stored.values())
    overlaps = {key: len(all_matches[key]) for key in KEY_NAMES}
    violations = []
    if spec.purpose == "independent_test":
        if not spec.independent or not spec.provenance.strip() or not samples:
            violations.append("missing_independence_evidence")
        if not training_present:
            violations.append("missing_training_lineage")
        if any(coverage[key] != len(samples) for key in ("exact", "template")):
            violations.append("incomplete_evaluation_fingerprint_coverage")
        if any(entry["coverage"][key] != entry["count"] for entry in stored.values() for key in ("exact", "template")):
            violations.append("unverified_lineage_fingerprint_coverage")
        if any(overlaps.values()):
            violations.append("lineage_overlap")
    elif spec.purpose == "development" and any(
        any(entry["overlaps"].values()) for entry in entries if entry["purpose"] == "train"
    ):
        violations.append("training_development_overlap")
    eligible = not violations
    return {
        "purpose": spec.purpose, "count": len(samples),
        "lineage_available": lineage is not None,
        "training_lineage_present": training_present,
        "fingerprint_coverage": coverage,
        "entries": entries, "overlaps": overlaps,
        "violations": violations, "eligible": eligible,
        "independent_test_eligible": spec.purpose == "independent_test" and eligible,
        "count_definition": "Distinct matching hashed keys, unioned across lineage entries",
    }


def validate_lineage_report(report: dict[str, Any]) -> None:
    if not report["eligible"]:
        raise ValueError("Evaluation data is ineligible: " + ", ".join(report["violations"]))
