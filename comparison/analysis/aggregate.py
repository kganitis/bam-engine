from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_results(raw_dir) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(raw_dir).glob("*.json")):
        r = json.loads(f.read_text())
        t, p, pop = r.get("timing", {}), r.get("process", {}), r.get("population", {})
        rid = r.get("run_id", f.stem)
        rep = int(rid.split("rep")[-1]) if "rep" in rid else 0
        rows.append(
            {
                "run_id": rid,
                "framework": r.get("framework"),
                "status": r.get("status"),
                "n_firms": pop.get("n_firms"),
                "n_agents_total": pop.get("n_agents_total"),
                "rep": rep,
                "steady_per_period": t.get("steady_state_per_period_seconds"),
                "throughput": t.get("throughput_agent_steps_per_s"),
                "peak_rss_bytes": p.get("peak_rss_bytes"),
                "startup_seconds": p.get("startup_seconds"),
            }
        )
    return pd.DataFrame(rows)
