import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_paths(cfg: dict, base_dir: str) -> dict:
    """Resolve relative paths in config relative to base_dir."""
    base = Path(base_dir)
    cfg["data"]["input_csv"] = str(base / cfg["data"]["input_csv"])
    cfg["embedding"]["cache_path"] = str(base / cfg["embedding"]["cache_path"])
    return cfg
