"""ASV benchmarks for BAM Engine simulation performance.

These benchmarks track performance across commits with machine-specific
baselines, solving the CI runner variability issue.
"""

import bamengine as bam


class SimulationSuite:
    """Benchmark suite for full simulation runs."""

    # ASV timeout (seconds) - large/1000 periods needs ~80-90s
    timeout = 120

    # Configuration parameters
    params = ["small", "medium", "large"]
    param_names = ["config"]

    # Configuration details
    configs = {
        "small": {"n_firms": 100, "n_households": 500, "n_banks": 10},
        "medium": {"n_firms": 200, "n_households": 1000, "n_banks": 10},
        "large": {"n_firms": 500, "n_households": 2500, "n_banks": 10},
    }

    def setup(self, config):
        """Setup simulation before each benchmark run."""
        cfg = self.configs[config]
        # Disable logging for performance benchmarks
        self.sim = bam.Simulation.init(
            n_firms=cfg["n_firms"],
            n_households=cfg["n_households"],
            n_banks=cfg["n_banks"],
            seed=42,
            logging={"default_level": "ERROR"},
        )

    def time_simulation_100_periods(self, config):
        """Benchmark 100 simulation periods."""
        self.sim.run(n_periods=100)

    def time_simulation_1000_periods(self, config):
        """Benchmark 1000 simulation periods (full baseline)."""
        self.sim.run(n_periods=1000)


class PipelineSuite:
    """Benchmark suite for pipeline operations."""

    def setup(self):
        """Setup minimal simulation."""
        self.sim = bam.Simulation.init(
            n_firms=100,
            n_households=500,
            n_banks=10,
            seed=42,
            logging={"default_level": "ERROR"},
        )

    def time_single_step(self):
        """Benchmark a single simulation step (all 39 events)."""
        self.sim.step()


class MemorySuite:
    """Benchmark memory usage."""

    params = [100, 200, 500]
    param_names = ["n_firms"]

    def peakmem_simulation_init(self, n_firms):
        """Peak memory during simulation initialization."""
        sim = bam.Simulation.init(
            n_firms=n_firms,
            n_households=n_firms * 5,
            n_banks=max(2, n_firms // 10),
            seed=42,
            logging={"default_level": "ERROR"},
        )
        return sim

    def peakmem_simulation_100_periods(self, n_firms):
        """Peak memory during 100-period simulation."""
        sim = bam.Simulation.init(
            n_firms=n_firms,
            n_households=n_firms * 5,
            n_banks=max(2, n_firms // 10),
            seed=42,
            logging={"default_level": "ERROR"},
        )
        sim.run(n_periods=100)
