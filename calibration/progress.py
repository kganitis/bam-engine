"""
Progress Tracking
=================

Progress bar and statistics tracking for long-running calibration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Try to import tqdm for nicer progress bars
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@dataclass
class ProgressStats:
    """Statistics for progress tracking."""

    completed: int = 0
    total: int = 0
    start_time: float = 0.0
    best_score: float = float("inf")
    best_params: dict[str, Any] = field(default_factory=dict)
    stage: str = ""
    current_config: dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """
    Track and display progress for long-running calibration.

    Features:
    - Percentage complete with progress bar
    - Elapsed time and ETA
    - Processing rate (configs/minute)
    - Best score found so far
    - Current configuration being evaluated

    Parameters
    ----------
    total : int
        Total number of items to process.
    stage : str
        Name of the current stage (e.g., "Grid Search").
    use_tqdm : bool
        Whether to use tqdm if available.
    update_interval : int
        How often to update display (every N items).
    """

    def __init__(
        self,
        total: int,
        stage: str = "",
        use_tqdm: bool = True,
        update_interval: int = 1,
    ):
        self.stats = ProgressStats(
            total=total,
            start_time=time.time(),
            stage=stage,
        )
        self.update_interval = update_interval
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self._pbar: Any = None
        self._last_print_time = 0.0

        if self.use_tqdm:
            self._pbar = tqdm(
                total=total,
                desc=stage,
                unit="cfg",
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] Best: {postfix}"
                ),
            )
            self._pbar.set_postfix_str("score=-.--")

    def update(
        self,
        n: int = 1,
        score: float | None = None,
        params: dict[str, Any] | None = None,
        force_print: bool = False,
    ):
        """
        Update progress with optional score tracking.

        Parameters
        ----------
        n : int
            Number of items completed in this update.
        score : float, optional
            Score of the current configuration.
        params : dict, optional
            Parameters of the current configuration.
        force_print : bool
            Force a print update even if not at interval.
        """
        self.stats.completed += n

        # Track best score
        if score is not None and score < self.stats.best_score:
            self.stats.best_score = score
            self.stats.best_params = params or {}

        if params:
            self.stats.current_config = params

        if self.use_tqdm and self._pbar:
            self._pbar.update(n)
            if score is not None:
                self._pbar.set_postfix_str(f"score={self.stats.best_score:.2f}")
        else:
            # Fallback: print progress at intervals
            now = time.time()
            should_print = (
                force_print
                or self.stats.completed == self.stats.total
                or self.stats.completed % self.update_interval == 0
                or (now - self._last_print_time) >= 10.0  # At least every 10 seconds
            )
            if should_print:
                self._print_progress()
                self._last_print_time = now

    def _print_progress(self):
        """Print progress to console (fallback when tqdm unavailable)."""
        stats = self.stats
        elapsed = time.time() - stats.start_time
        pct = 100 * stats.completed / stats.total if stats.total > 0 else 0

        # Calculate rate and ETA
        rate = stats.completed / elapsed if elapsed > 0 else 0
        remaining = stats.total - stats.completed
        eta_seconds = remaining / rate if rate > 0 else 0

        # Format time strings
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta_seconds)

        # Build progress bar
        bar_width = 30
        filled = (
            int(bar_width * stats.completed / stats.total) if stats.total > 0 else 0
        )
        bar = "█" * filled + "░" * (bar_width - filled)

        # Build status line
        status = (
            f"\r[{stats.stage}] {bar} {pct:5.1f}% "
            f"({stats.completed}/{stats.total}) | "
            f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
            f"Rate: {rate * 60:.1f} cfg/min | "
            f"Best: {stats.best_score:.2f}"
        )

        # Print with carriage return for in-place update
        print(status, end="", flush=True)

        # Print newline when complete
        if stats.completed >= stats.total:
            print()

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def close(self):
        """Close progress bar and print summary."""
        if self.use_tqdm and self._pbar:
            self._pbar.close()

        # Print final summary
        elapsed = time.time() - self.stats.start_time
        print(f"\n{self.stats.stage} completed:")
        print(f"  Total configs: {self.stats.completed}")
        print(f"  Total time: {self._format_time(elapsed)}")
        if elapsed > 0:
            print(f"  Avg rate: {self.stats.completed / elapsed * 60:.1f} configs/min")
        if self.stats.best_score < float("inf"):
            print(f"  Best score: {self.stats.best_score:.2f}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
