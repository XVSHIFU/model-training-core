"""Run portable engineering acceptance using only public synthetic fixtures."""
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
from training_core.runner import code_fingerprint


def main():
    run = ROOT / "runs" / ("engineering-" + uuid.uuid4().hex[:10])
    run.mkdir(parents=True, exist_ok=False)
    before = code_fingerprint()
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml", str(run / "pytest.xml")],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (run / "pytest.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    suites = ET.parse(run / "pytest.xml").getroot().findall("testsuite")
    counts = {name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
              for name in ("tests", "failures", "errors", "skipped")}
    entrypoint = Path(sys.executable).parent / ("model-workflow.exe" if sys.platform == "win32" else "model-workflow")
    cli = subprocess.run([str(entrypoint), "--help"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    report = {
        "status": "passed" if result.returncode == 0 and cli.returncode == 0 and before == code_fingerprint() else "failed",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Engineering acceptance with synthetic fixtures; no business generalization claim",
        "pytest": {**counts, "passed": counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"], "exit_code": result.returncode},
        "installed_entrypoint": {"name": "model-workflow", "exit_code": cli.returncode},
        "runtime": runtime_versions(),
        "source_sha256": before["sha256"],
        "covered": ["real three-class training and reload", "legacy source tests", "portable artifact lineage",
                    "dynamic development-use exclusion", "malformed and corrupted artifact audit",
                    "manifest-directory output protection", "no task imports in core"],
    }
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (run / "summary.json").write_text(content, encoding="utf-8")
    (ROOT / "docs" / "engineering-verification.json").write_text(content, encoding="utf-8")
    print(content)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
