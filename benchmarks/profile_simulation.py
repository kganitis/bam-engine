"""Profile simulation performance using cProfile.

Identifies performance bottlenecks by profiling function-level execution times.
"""

import cProfile
import pstats
import logging
from bamengine import Simulation

# Disable logging for profiling
logging.getLogger('bamengine').setLevel(logging.ERROR)


def profile_full_simulation():
    """Profile a full simulation run."""
    sim = Simulation.init(n_firms=100, n_households=500, seed=42)
    sim.run(100)


if __name__ == "__main__":
    print("Profiling BAM Engine simulation...")
    print("Configuration: 100 firms, 500 households, 100 periods")
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

    # Save for later analysis
    stats.dump_stats('simulation_profile.prof')
    print()
    print("Profile saved to: simulation_profile.prof")
    print("Analyze with: snakeviz simulation_profile.prof")
