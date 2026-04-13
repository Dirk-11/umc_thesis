"""
Shared utilities: config loading, path resolution, logging setup.

Every script imports from here so paths and config are consistent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML config from disk.

    Looks for config.yaml in the current directory first, then walks up
    to find it so scripts can be run from anywhere in the project.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        # Walk up from cwd looking for the config
        current = Path.cwd()
        for parent in [current, *current.parents]:
            candidate = parent / config_path
            if candidate.exists():
                config_path = candidate
                break
        else:
            raise FileNotFoundError(f"Could not find {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_project_root(config: dict[str, Any]) -> Path:
    """Return the project root as an absolute Path."""
    return Path(config["project"]["root"]).expanduser().resolve()


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path from config['paths'][key] relative to project root."""
    root = get_project_root(config)
    rel = config["paths"][key]
    return (root / rel).resolve()


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
