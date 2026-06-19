from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _median(df):
    ok = df[df["status"] == "ok"]
    return ok.groupby(["framework", "n_agents_total"], as_index=False)[
        ["steady_per_period", "peak_rss_bytes", "throughput"]
    ].median()


def scaling_curve(df, out):
    out = Path(out)
    m = _median(df)
    fig, ax = plt.subplots()
    for fw, g in m.groupby("framework"):
        g = g.sort_values("n_agents_total")
        ax.plot(g["n_agents_total"], g["steady_per_period"], marker="o", label=fw)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("agents")
    ax.set_ylabel("seconds per period (median)")
    ax.set_title("Steady-state per-period time vs population")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def memory_curve(df, out):
    out = Path(out)
    m = _median(df)
    fig, ax = plt.subplots()
    for fw, g in m.groupby("framework"):
        g = g.sort_values("n_agents_total")
        ax.plot(g["n_agents_total"], g["peak_rss_bytes"] / 1e6, marker="s", label=fw)
    ax.set_xscale("log")
    ax.set_xlabel("agents")
    ax.set_ylabel("peak RSS (MB)")
    ax.set_title("Peak memory vs population")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def speedup_bar(df, out):
    out = Path(out)
    m = _median(df)
    base = m[m["framework"] == "bamengine"].set_index("n_agents_total")[
        "steady_per_period"
    ]
    fig, ax = plt.subplots()
    for fw, g in m[m["framework"] != "bamengine"].groupby("framework"):
        g = g.sort_values("n_agents_total")
        slowdowns = []
        for n, row_val in zip(
            g["n_agents_total"], g["steady_per_period"], strict=False
        ):
            bam_val = base.get(n, float("nan"))
            if math.isnan(float(bam_val)) or bam_val == 0:
                slowdowns.append(float("nan"))
            else:
                slowdowns.append(row_val / bam_val)
        ax.plot(g["n_agents_total"], slowdowns, marker="^", label=f"{fw} / bamengine")
    ax.set_xscale("log")
    ax.set_xlabel("agents")
    ax.set_ylabel("slowdown vs bamengine (x)")
    ax.set_title("Relative slowdown vs bamengine")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
