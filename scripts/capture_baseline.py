"""Read-only legacy baseline. Private inputs/results stay in ignored runs/."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda item: item.item()) + "\n", encoding="utf-8")


def upload_fixture():
    rows = []
    for label, body in [("confirmed_upload_success", '{"success":true,"url":"/uploads/%s.txt"}'),
                        ("failed", '{"success":false,"message":"file type rejected %s"}')]:
        for i in range(24):
            rows.append({"event_id": f"fixture_{label}_{i}", "label": label,
                         "request": {"method": "POST", "uri": "/api/upload", "filename": f"sample{i}.txt", "file_ext": "txt", "body_excerpt": "synthetic fixture"},
                         "response": {"status_code": 200, "content_type": "application/json", "body": body % i}})
    rows.append({"event_id": "fixture_unknown", "label": "unknown", "request": {"method": "POST", "uri": "/api/upload"}, "response": {"status_code": 200, "body": "OK"}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.legacy.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Baseline outputs must be outside the legacy directory")
    output.mkdir(parents=True, exist_ok=False)
    protected = {}
    for directory in ("src", "tests", "pipelines", "scripts", "data", "models", "reports"):
        for path in (source / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                protected[path.relative_to(source).as_posix()] = digest(path)
    for name in ("pyproject.toml", "uv.lock", "README.md", "USAGE.md"):
        protected[name] = digest(source / name)
    write(output / "protected_assets.json", protected)
    with zipfile.ZipFile(output / "legacy_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in protected:
            if relative.endswith(".py") or relative in {"pyproject.toml", "uv.lock"}:
                archive.write(source / relative, relative)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([str(source / "src"), str(source)]))
    test = subprocess.run([sys.executable, "-m", "pytest", str(source / "tests"), "-q", "-p", "no:cacheprovider", "--basetemp", str(output / "pytest-temp")], cwd=source, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (output / "pytest.log").write_text(test.stdout + test.stderr, encoding="utf-8")
    if test.returncode:
        raise RuntimeError(f"Legacy baseline tests failed: {output / 'pytest.log'}")
    sys.path.insert(0, str(source / "src"))
    sys.dont_write_bytecode = True
    from upload_judge.ml_model import UploadClassifier
    from upload_judge.judge import UploadJudge
    from upload_judge.schemas import UploadEvent
    from upload_judge.features import build_model_text

    rows = [json.loads(line) for line in (source / "data/test-v4/test_cases_and_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    events = [UploadEvent.model_validate(row["event"]) for row in rows]
    model = UploadClassifier.load(source / "models/champion_v4.joblib")
    results = [item.to_dict() for item in UploadJudge(classifier=model).judge_batch(events)]
    (output / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    fixture = upload_fixture()
    write(output / "upload_fixture.json", fixture)
    fixture_events = [UploadEvent.model_validate(row) for row in fixture]
    training = {}
    for backend in ("logistic", "lightgbm"):
        candidate = UploadClassifier(backend=backend)
        manifest = candidate.fit(fixture_events, [row["label"] for row in fixture])
        candidate.save(output / f"fixture_{backend}.joblib")
        training[backend] = {"manifest": manifest, "params": candidate.classifier.get_params(), "vocabulary": candidate.vectorizer.vocabulary_, "probabilities": candidate.predict_success_proba(fixture_events).tolist()}
    write(output / "training_reference.json", {"texts": [build_model_text(e) for e in fixture_events], "structured": [UploadClassifier._struct_row(e) for e in fixture_events], "backends": training})
    summary = {"status": "passed", "purpose": "engineering_baseline", "legacy": str(source), "predictions": len(results), "protected_files": len(protected), "pytest": test.stdout.strip().splitlines()[-1], "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn", "lightgbm", "pydantic", "pytest")}}
    write(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
