"""Build and exercise a real wheel in a new isolated environment.

Requires uv on PATH. Uses locked dependencies and public synthetic fixtures,
keeps the existing environment intact, and retains all logs under runs/.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verify_engineering import engineering_inputs


def main():
    run = ROOT / "runs" / ("wheel-" + uuid.uuid4().hex[:10])
    run.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "MODEL_TRAINING_HOME", "MODEL_TRAINING_PROTECTED_PATHS"):
        env.pop(name, None)
    report = {"format_version": 1, "status": "failed", "verified_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Real wheel installation and synthetic CLI integration; no business generalization claim",
              "run_dir": str(run), "steps": []}

    def command(name, args, cwd=ROOT, process_env=None, json_result=False):
        step = {"name": name, "command": [str(arg) for arg in args], "log": name + ".log"}
        report["steps"].append(step)
        try:
            result = subprocess.run(step["command"], cwd=cwd, env=process_env or env,
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            step.update(exit_code=None, error_type=type(exc).__name__, error=str(exc))
            (run / step["log"]).write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            raise
        step["exit_code"] = result.returncode
        (run / step["log"]).write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} exited with {result.returncode}; see {step['log']}")
        return json.loads(result.stdout) if json_result else result.stdout

    try:
        report["inputs_before"] = engineering_inputs(ROOT)
        command("build", ["uv", "build", "--wheel", "--out-dir", run / "wheels"])
        wheels = list((run / "wheels").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("Expected exactly one built wheel")
        wheel = wheels[0]
        report["wheel"] = {"name": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
        command("create_environment", ["uv", "venv", "--python", sys.executable, run / "env"])
        executable_dir = run / "env" / ("Scripts" if os.name == "nt" else "bin")
        python = executable_dir / ("python.exe" if os.name == "nt" else "python")
        cli = executable_dir / ("model-workflow.exe" if os.name == "nt" else "model-workflow")
        report["interpreter"] = str(python)
        requirements = run / "requirements.txt"
        command("export_lock", ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--format", "requirements-txt", "--output-file", requirements])
        command("install_dependencies", ["uv", "pip", "install", "--python", python, "--require-hashes", "--requirements", requirements])
        command("install_wheel", ["uv", "pip", "install", "--python", python, "--no-deps", wheel])
        client = run / "client"
        client.mkdir()
        for name in ("text_train.jsonl", "text_dev.jsonl"):
            shutil.copyfile(ROOT / "examples" / name, client / name)
        config = json.loads((ROOT / "configs" / "text_demo.json").read_text(encoding="utf-8"))
        config["datasets"]["train"]["path"] = "text_train.jsonl"
        config["datasets"]["development"]["path"] = "text_dev.jsonl"
        config["run_root"] = "runs"
        config_path = client / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        report["installed"] = command("installation_identity", [python, "-I", "-c", (
            "import json, pathlib, sys; import training_core.runner as r; "
            "from training_core.artifacts import runtime_versions; "
            "print(json.dumps({'module':r.__file__, 'prefix':sys.prefix, 'runtime':runtime_versions(), 'code':r.code_fingerprint()}))"
        )], cwd=client, json_result=True)
        installed = report["installed"]
        if not Path(installed["module"]).resolve().is_relative_to((run / "env").resolve()):
            raise ValueError("Import did not resolve to the isolated wheel installation")
        if installed["code"] != report["inputs_before"]["source"]:
            raise ValueError("Installed wheel source identity differs from the checkout")
        report["project_only_fingerprint"] = all(
            name.split("/")[0] in {"training_core", "training_backends", "training_tasks", "upload_judge"}
            for name in installed["code"]["files"])
        if not report["project_only_fingerprint"]:
            raise ValueError("Source fingerprint includes unrelated dependency files")
        command("entrypoint_help", [cli, "--help"], cwd=client)
        report["validation"] = command("validate", [cli, "validate-data", "--config", config_path], cwd=client, json_result=True)
        report["training"] = command("train", [cli, "train", "--config", config_path], cwd=client, json_result=True)
        artifact = report["training"]["artifact"]
        report["prediction"] = command("predict", [cli, "predict", "--artifact", artifact, "--input", "text_dev.jsonl"], cwd=client, json_result=True)
        report["evaluation"] = command("evaluate", [cli, "evaluate", "--artifact", artifact, "--data", "text_dev.jsonl", "--purpose", "development", "--provenance", "public synthetic wheel smoke fixture"], cwd=client, json_result=True)
        for key in ("validation", "training", "prediction", "evaluation"):
            if Path(report[key]["run_dir"]).resolve().parent != (client / "runs").resolve():
                raise ValueError(f"{key} output escaped the client workspace")
        home = client / "explicit-home"
        home_env = {**env, "MODEL_TRAINING_HOME": str(home)}
        report["home_prediction"] = command("predict_home", [cli, "predict", "--artifact", artifact, "--input", "text_dev.jsonl"],
                                            cwd=client, process_env=home_env, json_result=True)
        if Path(report["home_prediction"]["run_dir"]).resolve().parent != (home / "runs").resolve():
            raise ValueError("MODEL_TRAINING_HOME was not used for default output")
        # Change a package outside this distribution in the disposable installation.
        # This is stronger than merely inspecting a reported scope string.
        sentinel = Path(installed["module"]).parent.parent / "unrelated_wheel_check.py"
        sentinel.write_text("# unrelated dependency probe\n", encoding="utf-8")
        changed = command("dependency_exclusion", [python, "-I", "-c", "import json; from training_core.provenance import code_fingerprint; print(json.dumps(code_fingerprint()))"], cwd=client, json_result=True)
        report["unrelated_file_does_not_change_fingerprint"] = changed == installed["code"]
        if changed != installed["code"]:
            raise ValueError("Unrelated installed source changed the project fingerprint")
        report["inputs_after"] = engineering_inputs(ROOT)
        report["inputs_unchanged"] = report["inputs_before"] == report["inputs_after"]
        if not report["inputs_unchanged"]:
            raise ValueError("Checkout verification inputs changed while testing the wheel")
        report["status"] = "passed"
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (run / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(run / "summary.json")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
