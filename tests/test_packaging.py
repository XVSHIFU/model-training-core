"""Fast boundary checks; verify_wheel.py separately exercises a real installation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from training_core import paths, provenance

ROOT = Path(__file__).resolve().parents[1]


def load_verifier():
    spec = importlib.util.spec_from_file_location("engineering_verifier", ROOT / "scripts" / "verify_engineering.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workspace_follows_call_time_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_TRAINING_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    assert paths.workspace_root() == tmp_path
    assert paths.default_run_root() == tmp_path / "runs"
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert paths.workspace_root() == other


def test_workspace_home_overrides_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_TRAINING_HOME", "chosen-home")
    assert paths.workspace_root() == tmp_path / "chosen-home"
    assert paths.default_run_root() == tmp_path / "chosen-home" / "runs"
    assert not paths.workspace_root().exists()  # resolving paths has no write side effect


def test_code_identity_excludes_unrelated_packages(tmp_path, monkeypatch):
    for name in provenance.PACKAGE_NAMES:
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text("# project package\n", encoding="utf-8")
    monkeypatch.setattr(provenance, "__file__", str(tmp_path / "training_core" / "provenance.py"))
    initial = provenance.code_fingerprint()
    foreign = tmp_path / "foreign_dependency"
    foreign.mkdir()
    (foreign / "code.py").write_text("changed = True\n", encoding="utf-8")
    assert provenance.code_fingerprint() == initial
    assert len(initial["files"]) == 4
    (tmp_path / "training_core" / "__init__.py").write_text("# changed project code\n", encoding="utf-8")
    assert provenance.code_fingerprint()["sha256"] != initial["sha256"]


def test_incomplete_installation_never_hashes_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "__file__", str(tmp_path / "training_core" / "provenance.py"))
    with pytest.raises(RuntimeError, match="Incomplete project installation"):
        provenance.code_fingerprint()


def test_verification_identity_includes_test_config_and_fixture_changes(tmp_path, monkeypatch):
    verifier = load_verifier()
    monkeypatch.setattr(verifier, "code_fingerprint", lambda: {"sha256": "source", "files": {}})
    for name in ("tests", "configs", "examples", "scripts"):
        (tmp_path / name).mkdir()
    for name in ("pyproject.toml", "uv.lock", "tests/test_sample.py", "configs/demo.json",
                 "examples/sample.jsonl", "scripts/verify_engineering.py"):
        (tmp_path / name).write_text("first", encoding="utf-8")
    before = verifier.engineering_inputs(tmp_path)
    (tmp_path / "tests/test_sample.py").write_text("new expectation", encoding="utf-8")
    (tmp_path / "configs/demo.json").write_text("new config", encoding="utf-8")
    (tmp_path / "examples/sample.jsonl").write_text("new fixture", encoding="utf-8")
    after = verifier.engineering_inputs(tmp_path)
    assert all(before[name] != after[name] for name in ("tests", "configs", "fixtures"))
    assert before["source"] == after["source"]
    (tmp_path / "configs/private.local.json").write_text("private", encoding="utf-8")
    assert verifier.engineering_inputs(tmp_path) == after


@pytest.mark.parametrize("condition", ["missing_xml", "malformed_xml", "empty_xml", "missing_command"])
def test_failed_verification_always_retains_report(condition, tmp_path, monkeypatch, capsys):
    verifier = load_verifier()
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "engineering_inputs", lambda root: {"source": {"sha256": "fixed"}})
    monkeypatch.setattr(verifier, "runtime_versions", lambda: {"python": "test"})
    docs = tmp_path / "docs"
    docs.mkdir()
    historical = docs / "engineering-verification.json"
    historical.write_text("historical evidence", encoding="utf-8")

    def fake_run(command, **kwargs):
        if condition == "missing_command":
            raise FileNotFoundError("command unavailable")
        if "--junitxml" in command and condition != "missing_xml":
            xml = Path(command[command.index("--junitxml") + 1])
            xml.write_text("<" if condition == "malformed_xml" else "<testsuites/>", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(verifier.subprocess, "run", fake_run)
    assert verifier.main() == 1
    result = json.loads(capsys.readouterr().out)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["pytest"]["tests"] is None
    assert historical.read_text(encoding="utf-8") == "historical evidence"


def test_two_verifications_preserve_distinct_reports(tmp_path, monkeypatch, capsys):
    verifier = load_verifier()
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "engineering_inputs", lambda root: {"source": {"sha256": "fixed"}})
    monkeypatch.setattr(verifier, "runtime_versions", lambda: {"python": "test"})

    def fake_run(command, **kwargs):
        if "--junitxml" in command:
            Path(command[-1]).write_text('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"/></testsuites>', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(verifier.subprocess, "run", fake_run)
    assert verifier.main() == 0
    first = json.loads(capsys.readouterr().out)
    contents = Path(first["report"]).read_bytes()
    assert verifier.main() == 0
    second = json.loads(capsys.readouterr().out)
    assert first["report"] != second["report"]
    assert Path(first["report"]).read_bytes() == contents


def test_verification_detects_mid_run_input_changes(tmp_path, monkeypatch, capsys):
    verifier = load_verifier()
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    snapshots = iter([{"source": {"sha256": "before"}}, {"source": {"sha256": "after"}}])
    monkeypatch.setattr(verifier, "engineering_inputs", lambda root: next(snapshots))
    monkeypatch.setattr(verifier, "runtime_versions", lambda: {})

    def fake_run(command, **kwargs):
        if "--junitxml" in command:
            Path(command[-1]).write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(verifier.subprocess, "run", fake_run)
    assert verifier.main() == 1
    result = json.loads(capsys.readouterr().out)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert report["pytest"]["passed"] == 1
    assert not report["inputs_unchanged"]
