"""Small I/O helpers shared by scripts."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np


def _default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(json.JSONEncoder(indent=2, default=_default).encode(obj) + "\n")


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def git_rev() -> str:
    """Short commit of the code being run: from git, else from a GIT_COMMIT file at the repo root (written
    into code bundles that have no .git, e.g. Colab and laptop runs), else "unknown"."""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout.strip()
        if rev:
            return rev
    except Exception:
        pass
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GIT_COMMIT")
    if os.path.exists(f):
        return open(f).read().strip()
    return "unknown"


def provenance(script: str, **kw) -> dict:
    """Header stored next to every result file: which script and commit produced it."""
    import platform
    import time

    import torch

    return {"script": script, "git": git_rev(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.node(), "torch": torch.__version__, "threads": torch.get_num_threads(),
            "cuda": torch.cuda.is_available(), **kw}
