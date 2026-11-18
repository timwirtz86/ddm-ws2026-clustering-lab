"""Save and load intermediate results between the notebooks.

Each notebook builds on the previous one: ``02_distance`` saves the feature
matrix ``X`` and the distance matrix ``D``, ``03_clustering`` loads them. The
files live in ``data/processed/`` (not tracked by git) and are named
``<dataset>_<name>.npy`` (arrays) or ``<dataset>_<name>.json`` (everything
else).

If a file is missing, the ``load_*`` functions return ``None`` and the notebook
falls back to a simple baseline, so a later notebook never depends on an
earlier one being fully solved.

The folder can be redirected with the environment variable
``DDM_PROCESSED_DIR`` (used by the instructor to run the solutions without
touching your files).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def processed_dir() -> Path:
    """Folder for intermediate results (created on first save)."""
    return Path(os.environ.get("DDM_PROCESSED_DIR", _DEFAULT_DIR))


def _path(dataset: str, name: str, suffix: str) -> Path:
    return processed_dir() / f"{dataset}_{name}{suffix}"


def save_array(dataset: str, name: str, array) -> Path:
    """Save a numpy array as ``data/processed/<dataset>_<name>.npy``."""
    path = _path(dataset, name, ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array))
    return path


def load_array(dataset: str, name: str) -> np.ndarray | None:
    """Load ``<dataset>_<name>.npy``, or return ``None`` if it does not exist."""
    path = _path(dataset, name, ".npy")
    return np.load(path) if path.exists() else None


def _to_plain(obj):
    """Convert numpy scalars/arrays to plain Python types for JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_plain(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(dataset: str, name: str, obj) -> Path:
    """Save a dict/list as ``data/processed/<dataset>_<name>.json``."""
    path = _path(dataset, name, ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_plain(obj), indent=2), encoding="utf-8")
    return path


def load_json(dataset: str, name: str):
    """Load ``<dataset>_<name>.json``, or return ``None`` if it does not exist."""
    path = _path(dataset, name, ".json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
