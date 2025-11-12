"""Profile simulation performance using cProfile.

Identifies performance bottlenecks by profiling function-level execution times.
"""

import cProfile
import pstats
import logging
from pathlib import Path
from bamengine import Simulation

# Disable logging for profiling
logging.getLogger('bamengine').setLevel(logging.ERROR)


def profile_full_simulation():
    """Profile a full simulation run."""
    # Using 1000 periods as recommended in original BAM paper
    sim = Simulation.init(n_firms=100, n_households=500, seed=42)
    sim.run(1000)


if __name__ == "__main__":
    print("Profiling BAM Engine simulation...")
    print("Configuration: 100 firms, 500 households, 1000 periods")
    print()

    profiler = cProfile.Profile()
    profiler.enable()

    profile_full_simulation()

    profiler.disable()

    # Print stats
    print("=" * 70)
    print("Top 30 functions by cumulative time:")
    print("=" * 70)
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(30)

    print()
    print("=" * 70)
    print("Top 30 functions by total time (self):")
    print("=" * 70)
    stats.sort_stats('tottime')
    stats.print_stats(30)

    # Save for later analysis in benchmarks directory
    output_dir = Path(__file__).parent
    output_path = output_dir / "simulation_profile.prof"
    stats.dump_stats(str(output_path))
    print()
    print(f"Profile saved to: {output_path}")
    print(f"Analyze with: snakeviz {output_path}")
