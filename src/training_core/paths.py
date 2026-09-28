"""Resolve user workspaces independently of source or installation locations."""
from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    """Use an explicit home or the call-time cwd; never an installed package path."""
    configured = os.environ.get("MODEL_TRAINING_HOME", "")
    return Path(configured).expanduser().resolve() if configured.strip() else Path.cwd().resolve()


def default_run_root() -> Path:
    return workspace_root() / "runs"
