"""Config, seeding, and run directories."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(os.environ.get("UNL_ROOT", Path(__file__).resolve().parents[1]))
RUNS = Path(os.environ.get("UNL_RUNS", ROOT / "runs"))


def _merge(base: dict, child: dict) -> dict:
    out = dict(base)
    for k, v in child.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """YAML config (optional `base:` file merged underneath) with dotted `key=value` overrides."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        cfg = _merge(load_config(path.parent / cfg.pop("base")), cfg)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_dir(run_id: str, *parts: str) -> Path:
    path = RUNS / run_id / Path(*parts) if parts else RUNS / run_id
    if path.exists() and any(path.iterdir()) and not os.environ.get("UNL_RESUME"):
        raise FileExistsError(f"{path} exists; use a new run-id (set UNL_RESUME=1 to reuse)")
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


def append_jsonl(rows, path: Path) -> None:
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
