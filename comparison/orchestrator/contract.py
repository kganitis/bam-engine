from __future__ import annotations

import json
from dataclasses import asdict, dataclass

SCHEMA_VERSION = "1.0"
STATUS_OK, STATUS_ERROR, STATUS_SKIPPED, STATUS_TIMEOUT = (
    "ok",
    "error",
    "skipped",
    "timeout",
)

_TIMING_KEYS = (
    "init_seconds",
    "run_seconds",
    "steady_state_per_period_seconds",
    "throughput_agent_steps_per_s",
)


@dataclass
class RunRequest:
    """JSON contract for simulation run requests."""

    run_id: str
    framework: str
    model_params: dict
    population: dict
    n_periods: int
    warmup_periods: int
    seed: int
    collect_outputs: bool
    outputs_requested: list

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> RunRequest:
        """Deserialize from JSON string."""
        return cls(**json.loads(s))


@dataclass
class RunResult:
    """JSON contract for simulation run results."""

    schema_version: str
    run_id: str
    framework: str
    framework_version: str
    language: str
    language_version: str
    status: str
    error: object
    population: dict
    n_periods: int
    warmup_periods: int
    seed: int
    timing: dict
    outputs: object

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> RunResult:
        """Deserialize from JSON string."""
        return cls(**json.loads(s))

    def validate(self) -> list:
        """Validate result against schema. Returns list of problems (empty if valid)."""
        problems = []
        if self.schema_version != SCHEMA_VERSION:
            problems.append(f"schema_version {self.schema_version} != {SCHEMA_VERSION}")
        if self.status == STATUS_OK:
            for key in _TIMING_KEYS:
                if key not in self.timing:
                    problems.append(f"timing missing {key}")
        return problems
