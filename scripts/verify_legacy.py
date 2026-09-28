"""Verify refactor parity against a captured legacy baseline without modifying it.

Detailed diagnostics belong to ignored runs/. The public report contains only
aggregate results, never input records, prediction content, IDs or host paths.
Run with this repository's virtual environment and a trusted legacy model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_protected(source: Path, protected: dict[str, str]) -> dict[str, Any]:
    changed, missing = [], []
    for relative, expected in protected.items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Protected manifest entry escapes the legacy directory")
        if not path.is_file():
            missing.append(relative)
        elif digest(path) != expected:
            changed.append(relative)
    return {"checked": len(protected), "changed": changed, "missing": missing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=ROOT / "runs" / "baseline-verified")
    parser.add_argument("--output", type=Path, help="New private run directory; defaults to a unique runs/ directory")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "verification.json")
    args = parser.parse_args()
    source, baseline = args.legacy.resolve(), args.baseline.resolve()
    default_output = ROOT / "runs" / ("legacy-verification-" + uuid.uuid4().hex[:10])
    output, report_path = (args.output or default_output).resolve(), args.report.resolve()
    for destination in (output, report_path):
        if destination == source or destination.is_relative_to(source):
            raise ValueError("Verification outputs must be outside the legacy directory")
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Private diagnostics must remain inside the repository runs directory")
    if output == baseline or output.is_relative_to(baseline):
        raise ValueError("Verification must not overwrite its captured baseline")
    output.mkdir(parents=True, exist_ok=False)

    import joblib
    import numpy as np

    from training_core.contracts import TaskSpec
    from training_tasks.upload import UploadTask
    from training_tasks.upload.representation import represent

    baseline_summary = read_json(baseline / "summary.json")
    if baseline_summary.get("status") != "passed":
        raise ValueError("The captured legacy baseline did not pass")
    protected = read_json(baseline / "protected_assets.json")
    before = check_protected(source, protected)
    task = UploadTask()

    # Check preparation on the real frozen training file without fitting it.
    training_samples = [task.parse(row) for row in read_jsonl(source / "data" / "processed" / "train_v4.jsonl")]
    prepared_training, training_summary = task.prepare(training_samples)
    expected_training = {
        "total": 14105, "used": 13173, "excluded": 932,
        "reasons": {"target:unknown": 932}, "class_counts": {"0": 1083, "1": 12090},
    }
    preparation_exact = training_summary == expected_training and len(prepared_training) == 13173
    preparation_report = {
        **training_summary,
        "unknown_remains_labeled": all(sample.label_status == "labeled" for sample in training_samples if sample.target == "unknown"),
        "expected_counts_exact": preparation_exact,
    }
    del training_samples, prepared_training

    # Load the original artifact through the refactored compatibility class.
    cases = read_jsonl(source / "data" / "test-v4" / "test_cases_and_predictions.jsonl")
    samples = [task.parse(row["event"]) for row in cases]
    expected_predictions = read_jsonl(baseline / "predictions.jsonl")
    champion = task.load(source / "models" / "champion_v4.joblib")
    outputs = [prediction.output for prediction in task.predict(champion, samples)]
    unequal_indices = [index for index, (expected, actual) in enumerate(zip(expected_predictions, outputs)) if expected != actual]
    predictions_exact = len(outputs) == len(expected_predictions) and not unequal_indices

    fixture = read_json(baseline / "upload_fixture.json")
    fixture_samples = [task.parse(row) for row in fixture]
    fixture_events = [sample.inputs for sample in fixture_samples]
    reference = read_json(baseline / "training_reference.json")
    texts, structured = represent(fixture_events)
    text_match = texts == reference["texts"]
    structured_match = bool(np.array_equal(np.asarray(structured), np.asarray(reference["structured"])))
    backend_results = {}
    cli_results = {}
    for backend in ("logistic", "lightgbm"):
        expected = reference["backends"][backend]
        candidate = task.fit(fixture_samples, TaskSpec(task_id="upload", backend=backend, seed=42))
        saved_reference = joblib.load(baseline / f"fixture_{backend}.joblib")
        probabilities = candidate.predict_success_proba(fixture_events)
        expected_probabilities = np.asarray(expected["probabilities"], dtype=float)
        same_shape = probabilities.shape == expected_probabilities.shape
        probability_match = bool(np.array_equal(probabilities, expected_probabilities))
        maximum_difference = float(np.max(np.abs(probabilities - expected_probabilities))) if same_shape and probabilities.size else None
        manifest_match = {
            key: value for key, value in candidate.manifest.items() if key != "trained_at"
        } == {
            key: value for key, value in expected["manifest"].items() if key != "trained_at"
        }
        if backend == "logistic":
            learned_match = all(bool(np.array_equal(getattr(candidate.classifier, name), getattr(saved_reference["classifier"], name))) for name in ("coef_", "intercept_", "classes_", "n_iter_"))
        else:
            learned_match = candidate.classifier.booster_.dump_model() == saved_reference["classifier"].booster_.dump_model()
        artifact_path = output / f"fixture_{backend}.joblib"
        task.save(candidate, artifact_path)
        roundtrip = task.load(artifact_path)
        details = {
            "training_samples": candidate.manifest["training_samples"],
            "excluded_fixture_samples": len(fixture_samples) - candidate.manifest["training_samples"],
            "vocabulary_exact": candidate.vectorizer.vocabulary_ == expected["vocabulary"],
            "classifier_parameters_exact": candidate.classifier.get_params() == expected["params"],
            "vectorizer_parameters_exact": candidate.vectorizer.get_params() == saved_reference["vectorizer"].get_params(),
            "idf_exact": bool(np.array_equal(candidate.vectorizer.idf_, saved_reference["vectorizer"].idf_)),
            "learned_parameters_exact": learned_match,
            "manifest_except_training_time_exact": manifest_match,
            "probabilities_exact": probability_match,
            "maximum_probability_difference": maximum_difference,
            "joblib_roundtrip_exact": bool(np.array_equal(probabilities, roundtrip.predict_success_proba(fixture_events))),
        }
        details["status"] = "passed" if all(value for name, value in details.items() if name.endswith("_exact")) else "failed"
        backend_results[backend] = details

        # Exercise the real compatibility CLI in a separate Python process,
        # including argument parsing, model saving and its sidecar manifest.
        cli_path = output / f"legacy_cli_{backend}.joblib"
        cli = subprocess.run(
            [sys.executable, "-B", "-m", "upload_judge.cli", "train",
             "--data", str(baseline / "upload_fixture.json"), "--output", str(cli_path),
             "--model-type", "b1" if backend == "logistic" else "b2", "--seed", "42"],
            cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src")),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        (output / f"legacy_cli_{backend}.log").write_text(cli.stdout + cli.stderr, encoding="utf-8")
        cli_details = {"exit_code": cli.returncode, "status": "failed"}
        if cli.returncode == 0:
            cli_model = task.load(cli_path)
            cli_probabilities = cli_model.predict_success_proba(fixture_events)
            original_fixture_model = task.load(baseline / f"fixture_{backend}.joblib")
            cli_details.update({
                "training_samples": cli_model.manifest["training_samples"],
                "sidecar_manifest_created": cli_path.with_suffix(".manifest.json").is_file(),
                "sidecar_manifest_exact": read_json(cli_path.with_suffix(".manifest.json")) == cli_model.manifest,
                "vocabulary_exact": cli_model.vectorizer.vocabulary_ == expected["vocabulary"],
                "classifier_parameters_exact": cli_model.classifier.get_params() == expected["params"],
                "probabilities_exact": bool(np.array_equal(cli_probabilities, expected_probabilities)),
                "maximum_probability_difference": float(np.max(np.abs(cli_probabilities - expected_probabilities))),
                "complete_output_exact": [p.output for p in task.predict(cli_model, fixture_samples)] == [p.output for p in task.predict(original_fixture_model, fixture_samples)],
            })
            cli_details["status"] = "passed" if all(value for name, value in cli_details.items() if name.endswith("_exact")) and cli_details["sidecar_manifest_created"] else "failed"
        cli_results[backend] = cli_details

    after = check_protected(source, protected)
    protected_ok = not (before["changed"] or before["missing"] or after["changed"] or after["missing"])
    packages = {name: importlib.metadata.version(name) for name in baseline_summary["packages"]}
    packages_match = packages == baseline_summary["packages"]
    test_match = re.search(r"(\d+) passed", baseline_summary.get("pytest", ""))
    passed = (protected_ok and predictions_exact and text_match and structured_match
              and preparation_exact and preparation_report["unknown_remains_labeled"]
              and packages_match and all(value["status"] == "passed" for value in backend_results.values())
              and all(value["status"] == "passed" for value in cli_results.values()))
    report = {
        "status": "passed" if passed else "failed",
        "purpose": "engineering_refactor_parity_not_independent_model_generalization",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "captured_legacy_baseline": {
            "tests_passed": int(test_match.group(1)) if test_match else None,
            "predictions": baseline_summary["predictions"],
            "protected_files": baseline_summary["protected_files"],
        },
        "runtime_packages_match_baseline": packages_match,
        "legacy_training_data_preparation": preparation_report,
        "legacy_champion": {
            "predictions_checked": len(outputs),
            "reference_predictions": len(expected_predictions),
            "complete_output_exact": predictions_exact,
            "different_outputs": len(unequal_indices) + abs(len(outputs) - len(expected_predictions)),
        },
        "synthetic_fixture": {
            "samples": len(fixture_samples), "texts_exact": text_match,
            "structured_features": 24, "structured_values_exact": structured_match,
            "backends": backend_results,
        },
        "legacy_training_cli": {"synthetic_samples": len(fixture_samples), "backends": cli_results},
        "legacy_protected_assets": {
            "checked_before_and_after": len(protected),
            "changed_before": len(before["changed"]), "missing_before": len(before["missing"]),
            "changed_after": len(after["changed"]), "missing_after": len(after["missing"]),
            "all_sha256_unchanged": protected_ok,
        },
    }
    write_json(output / "diagnostics.json", {
        "prediction_difference_indices": unequal_indices,
        "protected_before": before, "protected_after": after,
        "runtime_packages": packages, "baseline_packages": baseline_summary["packages"],
    })
    write_json(output / "summary.json", report)
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
