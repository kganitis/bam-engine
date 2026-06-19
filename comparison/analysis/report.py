from __future__ import annotations

from pathlib import Path


def write_report(df, gate, env, out_path) -> Path:
    out = Path(out_path)
    lines = []
    lines.append("# Cross-Framework Benchmark Report\n")
    lines.append(f"Environment: {env.get('cpu')} ({env.get('n_cores')} cores)\n")
    lines.append("## Fidelity (behavioral-equivalence gate)\n")
    for fw, info in gate.get("frameworks", {}).items():
        status = "PASS" if info.get("passed") else "FAIL"
        block = "" if info.get("blocking", True) else " (non-blocking reference)"
        lines.append(f"- {fw}: {status}{block}")
    lines.append("\n## Steady-state per-period time (median seconds)\n")
    ok = df[df["status"] == "ok"]
    if not ok.empty:
        piv = ok.pivot_table(
            index="n_agents_total",
            columns="framework",
            values="steady_per_period",
            aggfunc="median",
        )
        lines.append(piv.to_markdown())
    lines.append("\n## Notes\n")
    lines.append("- Single-thread pinned. Timing runs serial; gate runs parallel.")
    lines.append("- Adaptive cap: see skips.json for sizes dropped per framework.")
    out.write_text("\n".join(lines))
    return out
