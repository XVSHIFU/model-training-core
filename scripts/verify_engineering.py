"""Verify the checkout; retain every success or failure in a new runs directory.

This checks editable/source engineering behavior. Wheel installation is a separate
verification layer. No existing docs report or earlier run is overwritten.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_core.artifacts import runtime_versions
from training_core.provenance import code_fingerprint, fingerprint_files


def engineering_inputs(root: Path) -> dict:
    """Identify code and the exact tests, fixtures and configuration being checked."""
    groups = {
        "tests": sorted((root / "tests").rglob("*.py")),
        "configs": sorted(path for path in (root / "configs").rglob("*.json")
                          if not path.name.endswith(".local.json")),
        "fixtures": sorted(path for path in (root / "examples").rglob("*") if path.is_file()),
        "environment": [root / "pyproject.toml", root / "uv.lock"],
        "verification_scripts": sorted((root / "scripts").glob("*.py")),
    }
    result = {name: fingerprint_files(root, files) for name, files in groups.items()}
    result["source"] = code_fingerprint()
    return result


def run_command(command: list[str], root: Path, log: Path) -> dict:
    """Keep actionable diagnostics even if a tool cannot start or times out."""
    try:
        process = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=600)
        output = process.stdout + process.stderr
        result = {"exit_code": process.returncode}
    except (OSError, subprocess.SubprocessError) as exc:
        output = f"{type(exc).__name__}: {exc}\n"
        result = {"exit_code": None, "error_type": type(exc).__name__, "error": str(exc)}
    log.write_text(output, encoding="utf-8")
    return {"command": command, "log": log.name, **result}


def pytest_counts(path: Path) -> dict:
    try:
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
        if not suites:
            raise ValueError("JUnit report contains no test suites")
        counts = {name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
                  for name in ("tests", "failures", "errors", "skipped")}
        counts["passed"] = max(0, counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"])
        return counts
    except (OSError, ValueError, ET.ParseError) as exc:
        return {"tests": None, "passed": None, "failures": None, "errors": None, "skipped": None,
                "report_error": f"{type(exc).__name__}: {exc}"}


def main():
    run = ROOT / "runs" / ("engineering-" + uuid.uuid4().hex[:10])
    run.mkdir(parents=True, exist_ok=False)
    report = {
        "format_version": 2,
        "status": "failed",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Checkout engineering acceptance with public synthetic fixtures; not wheel installation or business generalization",
        "interpreter": str(Path(sys.executable).resolve()),
        "run_dir": str(run),
    }
    try:
        before = engineering_inputs(ROOT)
        report["inputs_before"] = before
        report["source_sha256"] = before["source"]["sha256"]
        pytest = run_command([sys.executable, "-m", "pytest", "-q", "--junitxml", str(run / "pytest.xml")],
                             ROOT, run / "pytest.log")
        report["pytest"] = {**pytest, **pytest_counts(run / "pytest.xml")}
        entrypoint = Path(sys.executable).parent / ("model-workflow.exe" if sys.platform == "win32" else "model-workflow")
        report["installed_entrypoint"] = {"name": "model-workflow",
            **run_command([str(entrypoint), "--help"], ROOT, run / "entrypoint.log")}
        report["runtime"] = runtime_versions()
        after = engineering_inputs(ROOT)
        report["inputs_after"] = after
        report["inputs_unchanged"] = before == after
        counts = report["pytest"]
        passed = (counts["exit_code"] == 0 and counts["tests"] is not None and counts["tests"] > 0
                  and counts["failures"] == 0 and counts["errors"] == 0
                  and report["installed_entrypoint"]["exit_code"] == 0 and before == after)
        report["status"] = "passed" if passed else "failed"
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (run / "summary.json").write_text(content, encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(run / "summary.json")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
