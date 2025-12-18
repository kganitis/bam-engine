"""
Checkpoint Management
=====================

Saves and resumes calibration progress to/from JSON files.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import FIXED_PARAMS

# Default output directory (inside calibration package)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


class CheckpointManager:
    """
    Manages checkpointing for calibration runs.

    Supports:
    - Saving results incrementally during calibration
    - Resuming from a previous run
    - Fast lookup of already-evaluated configurations
    - Atomic writes to prevent corruption

    Parameters
    ----------
    filepath : str or Path
        Path to checkpoint file.
    resume : bool
        If True, load existing checkpoint. If False, start fresh.
    """

    def __init__(self, filepath: str | Path, resume: bool = False):
        self.filepath = Path(filepath)
        self.resume = resume
        self.data = self._load_or_create()
        self._configs_evaluated: set[str] = set()
        self._rebuild_evaluated_set()

    def _load_or_create(self) -> dict:
        """Load existing checkpoint or create new one."""
        if self.resume and self.filepath.exists():
            print(f"Resuming from checkpoint: {self.filepath}")
            with open(self.filepath) as f:
                return json.load(f)

        if self.resume and not self.filepath.exists():
            print(f"WARNING: Checkpoint not found: {self.filepath}")
            print("Starting fresh calibration instead.")
        elif self.filepath.exists():
            print("Starting fresh (overwriting existing checkpoint)")

        return {
            "version": "2.0",
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_configs_evaluated": 0,
                "stage": "starting",
                "n_seeds": 3,
                "n_periods": 1000,
                "burn_in": 500,
            },
            "fixed_params": dict(FIXED_PARAMS),
            "grid_results": [],
            "top_configurations": [],
        }

    def _rebuild_evaluated_set(self):
        """Rebuild set of evaluated config hashes for fast lookup."""
        for result in self.data.get("grid_results", []):
            self._configs_evaluated.add(result["config_hash"])

    def config_hash(self, params: dict) -> str:
        """
        Generate deterministic hash for configuration.

        Parameters
        ----------
        params : dict
            Parameter configuration.

        Returns
        -------
        str
            16-character hex hash.
        """
        sorted_items = sorted((str(k), str(v)) for k, v in params.items())
        return hashlib.md5(str(sorted_items).encode()).hexdigest()[:16]

    def is_evaluated(self, params: dict) -> bool:
        """
        Check if configuration was already evaluated.

        Parameters
        ----------
        params : dict
            Parameter configuration.

        Returns
        -------
        bool
            True if already evaluated.
        """
        return self.config_hash(params) in self._configs_evaluated

    def add_result(
        self,
        params: dict,
        scores: dict,
        seed_scores: list[float],
    ):
        """
        Add evaluation result.

        Parameters
        ----------
        params : dict
            Parameter configuration.
        scores : dict
            Score components and total.
        seed_scores : list[float]
            Total scores from each seed.
        """
        result = {
            "config_hash": self.config_hash(params),
            "params": params,
            "scores": {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in scores.items()
            },
            "seed_scores": seed_scores,
            "timestamp": datetime.now().isoformat(),
        }
        self.data["grid_results"].append(result)
        self._configs_evaluated.add(result["config_hash"])
        self.data["metadata"]["total_configs_evaluated"] = len(
            self.data["grid_results"]
        )

    def set_metadata(self, key: str, value: Any):
        """Set metadata value."""
        self.data["metadata"][key] = value

    def save(self):
        """Save checkpoint to disk (atomic write)."""
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()

        # Ensure parent directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write via temp file
        temp_path = self.filepath.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        os.replace(temp_path, self.filepath)

    def should_checkpoint(self, interval: int = 100) -> bool:
        """
        Check if it's time to checkpoint.

        Parameters
        ----------
        interval : int
            Save every N configurations.

        Returns
        -------
        bool
            True if should save now.
        """
        return self.data["metadata"]["total_configs_evaluated"] % interval == 0

    def get_top_configs(self, n: int = 20) -> list[dict]:
        """
        Get top n configurations by total score.

        Parameters
        ----------
        n : int
            Number of configurations to return.

        Returns
        -------
        list[dict]
            Top configurations sorted by score (ascending).
        """
        sorted_results = sorted(
            self.data["grid_results"],
            key=lambda x: x["scores"]["total"],
        )
        return sorted_results[:n]

    def get_config_by_rank(self, rank: int) -> dict | None:
        """
        Get configuration by rank (1 = best).

        Parameters
        ----------
        rank : int
            Rank to retrieve.

        Returns
        -------
        dict or None
            Configuration if found.
        """
        top = self.get_top_configs(rank)
        if len(top) >= rank:
            return top[rank - 1]
        return None

    def print_summary(self):
        """Print checkpoint summary."""
        meta = self.data["metadata"]
        print("\n" + "=" * 70)
        print("CHECKPOINT SUMMARY")
        print("=" * 70)
        print(f"  File: {self.filepath}")
        print(f"  Created: {meta.get('created_at', 'N/A')}")
        print(f"  Last updated: {meta.get('last_updated', 'N/A')}")
        print(f"  Configurations evaluated: {meta.get('total_configs_evaluated', 0)}")

        top = self.get_top_configs(1)
        if top:
            print(f"  Best score: {top[0]['scores']['total']:.2f}")
        print("=" * 70)


def get_default_checkpoint_path() -> Path:
    """Get default checkpoint file path."""
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / "calibration_checkpoint.json"
