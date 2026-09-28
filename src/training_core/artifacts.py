"""Safe run directories and backend-neutral artifact manifests."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import uuid

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_paths() -> list[Path]:
    roots = [REPOSITORY_ROOT / name for name in ("models", "data", "reports", ".git", ".venv")]
    settings = REPOSITORY_ROOT / "local" / "protected.json"
    if settings.exists():
        roots.extend(Path(p) for p in json.loads(settings.read_text(encoding="utf-8-sig"))["paths"])
    roots.extend(Path(p) for p in os.environ.get("MODEL_TRAINING_PROTECTED_PATHS", "").split(os.pathsep) if p)
    return [p.resolve() for p in roots]


def guard_output(path: str | Path, extra_protected=(), *, allow_existing: bool = False) -> Path:
    resolved = Path(path).resolve()
    for protected in [*protected_paths(), *(Path(p).resolve() for p in extra_protected)]:
        if resolved == protected or protected in resolved.parents:
            raise ValueError(f"Refusing to write protected asset: {resolved}")
    if resolved.exists() and not allow_existing:
        raise FileExistsError(f"Refusing to overwrite existing output: {resolved}")
    return resolved


def write_json(path: Path, value, *, overwrite: bool = False, extra_protected=()) -> None:
    target = guard_output(path, extra_protected, allow_existing=overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Initial writes use exclusive creation, including when another process races us.
    with target.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def begin_run(root: Path, command: str, config: dict, extra_protected=()) -> Path:
    guard_output(root, extra_protected, allow_existing=True)
    run = root.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10])
    guard_output(run, extra_protected)
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    write_json(run / "run.json", {"status": "running", "command": command, "started_at": datetime.now(timezone.utc).isoformat()})
    return run


def finish_run(run: Path, *, status: str, details=None) -> None:
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    record.update(status=status, finished_at=datetime.now(timezone.utc).isoformat(), details=details or {})
    write_json(run / "run.json", record, overwrite=True)


def runtime_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("model-training-core", "numpy", "scipy", "scikit-learn", "lightgbm", "pydantic", "joblib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def inventory(path: Path, root: Path) -> dict[str, str]:
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    return {item.relative_to(root).as_posix(): sha256_file(item) for item in paths if item.is_file()}


def load_manifest(path: Path) -> tuple[dict, Path]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("Unsupported artifact manifest version")
    for key in ("task_id", "backend", "model_path", "config_path", "lineage_path"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"Invalid or missing artifact field: {key}")
    root = manifest_path.parent.resolve()
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Artifact has no integrity inventory")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("Invalid artifact integrity inventory entry")
        item = (root / relative).resolve()
        if root not in item.parents or not item.is_file() or sha256_file(item) != expected:
            raise ValueError(f"Artifact integrity check failed: {relative}")
    model_path = (root / manifest["model_path"]).resolve()
    if root not in model_path.parents or not model_path.exists():
        raise ValueError("Invalid artifact model path")
    required = inventory(model_path, root)
    if not required or any(files.get(name) != digest for name, digest in required.items()):
        raise ValueError("Model is not fully covered by artifact integrity inventory")
    for key in ("config_path", "lineage_path"):
        if manifest.get(key) not in files:
            raise ValueError(f"Missing inventoried {key}")
    return manifest, model_path
