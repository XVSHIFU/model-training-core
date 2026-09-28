"""Content identities with explicit scope, independent of unrelated dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

PACKAGE_NAMES = ("training_core", "training_backends", "training_tasks", "upload_judge")


def fingerprint_files(root: Path, paths: Iterable[Path]) -> dict:
    """Hash an explicitly selected set; relative names make the identity portable."""
    root = root.resolve()
    files = {}
    for path in sorted(paths):
        path = path.resolve()
        name = path.relative_to(root).as_posix()
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def code_fingerprint() -> dict:
    """Hash only this distribution's four Python packages without importing them.

    The packages are siblings in both the src layout and a wheel installation.
    Missing packages indicate an incomplete installation and are an error; never
    fall back to traversing the enclosing site-packages or environment directory.
    Dependency versions are separate environment provenance, not project source.
    """
    source = Path(__file__).resolve().parent.parent
    missing = [name for name in PACKAGE_NAMES if not (source / name / "__init__.py").is_file()]
    if missing:
        raise RuntimeError("Incomplete project installation; missing packages: " + ", ".join(missing))
    paths = [path for name in PACKAGE_NAMES for path in (source / name).rglob("*.py")]
    return {"scope": "project_package_python_sources_v1", **fingerprint_files(source, paths)}
