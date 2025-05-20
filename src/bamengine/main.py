"""Command‑line demo runner for BAM Engine."""

from __future__ import annotations

import argparse
import logging

from bamengine.scheduler import Scheduler


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run minimal BAM Engine demo.")
    p.add_argument("--firms", type=int, default=100, help="Number of emp")
    p.add_argument("--households", type=int, default=500, help="Number of households")
    p.add_argument("--banks", type=int, default=10, help="Number of banks")
    p.add_argument("--steps", type=int, default=50, help="Simulation periods")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--h-rho", type=float, default=0.10, help="Max quantity shock")
    return p.parse_args()


def main() -> None:
    args = _cli()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    sched = Scheduler.init(
        n_firms=args.firms,
        n_households=args.households,
        n_banks=args.banks,
        h_rho=args.h_rho,
        seed=args.seed,
    )

    for t in range(1, args.steps + 1):
        log.info("=== PERIOD %d ===", t)
        sched.step()

    log.info("Simulation finished.")


if __name__ == "__main__":
    main()
