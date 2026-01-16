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


class CriticalEventSuite:
    """Benchmark critical path events (goods/labor/credit markets).

    These events account for ~58% of total simulation time and are the
    primary bottlenecks identified in profiling (see docs/performance.rst).
    """

    timeout = 60

    def setup(self):
        """Setup simulation in steady state."""
        self.sim = bam.Simulation.init(
            n_firms=100,
            n_households=500,
            n_banks=10,
            seed=42,
            logging={"default_level": "ERROR"},
        )
        # Run a few periods to reach steady state
        self.sim.run(10)

    # Goods market events (~48% of runtime)
    def time_consumers_decide_firms_to_visit(self):
        """Benchmark firm selection for shopping."""
        event = self.sim.get_event("consumers_decide_firms_to_visit")
        event.execute(self.sim)

    def time_consumers_shop_sequential(self):
        """Benchmark sequential shopping (all rounds)."""
        event = self.sim.get_event("consumers_shop_sequential")
        event.execute(self.sim)

    # Labor market events (~5%)
    def time_workers_decide_firms_to_apply(self):
        """Benchmark firm selection for job applications."""
        event = self.sim.get_event("workers_decide_firms_to_apply")
        event.execute(self.sim)

    def time_firms_hire_workers(self):
        """Benchmark worker hiring process."""
        event = self.sim.get_event("firms_hire_workers")
        event.execute(self.sim)

    # Credit market events (~5%)
    def time_firms_send_one_loan_app(self):
        """Benchmark loan application submission."""
        event = self.sim.get_event("firms_send_one_loan_app")
        event.execute(self.sim)

    def time_banks_provide_loans(self):
        """Benchmark loan provision by banks."""
        event = self.sim.get_event("banks_provide_loans")
        event.execute(self.sim)


class InitSuite:
    """Benchmark initialization costs across scales."""

    timeout = 60
    params = [100, 200, 500, 1000]
    param_names = ["n_firms"]

    def time_simulation_init(self, n_firms):
        """Time simulation initialization."""
        bam.Simulation.init(
            n_firms=n_firms,
            n_households=n_firms * 5,
            n_banks=max(2, n_firms // 10),
            seed=42,
            logging={"default_level": "ERROR"},
        )


class LoanBookSuite:
    """Benchmark sparse relationship operations.

    The LoanBook uses COO sparse format for memory-efficient loan storage.
    These benchmarks track the performance of common operations.
    """

    timeout = 30

    def setup(self):
        """Setup LoanBook with pre-populated loans."""
        import numpy as np

        from bamengine.relationships import LoanBook

        self.np = np
        self.loans = LoanBook()
        self.rng = np.random.default_rng(42)
        # Pre-populate with 500 loans (50 per bank × 10 banks)
        for bank in range(10):
            n = 50
            self.loans.append_loans_for_lender(
                lender_idx=np.intp(bank),
                borrower_indices=self.rng.integers(0, 100, n, dtype=np.int64),
                amount=self.rng.uniform(100, 1000, n),
                rate=self.rng.uniform(0.02, 0.05, n),
            )

    def time_append_loans(self):
        """Time appending 100 loans."""
        self.loans.append_loans_for_lender(
            lender_idx=self.np.intp(0),
            borrower_indices=self.rng.integers(0, 100, 100, dtype=self.np.int64),
            amount=self.rng.uniform(100, 1000, 100),
            rate=self.rng.uniform(0.02, 0.05, 100),
        )

    def time_debt_per_borrower(self):
        """Time debt aggregation by borrower."""
        self.loans.debt_per_borrower(n_borrowers=100)

    def time_purge_borrowers(self):
        """Time purging 10 bankrupt borrowers."""
        bankrupt = self.rng.choice(100, 10, replace=False).astype(self.np.int64)
        self.loans.purge_borrowers(bankrupt)


class ScalingSuite:
    """Track performance scaling with agent count.

    Verifies that performance scales sub-linearly with agent count
    due to NumPy vectorization efficiency.
    """

    timeout = 120
    params = [50, 100, 200, 400]
    param_names = ["n_firms"]

    def setup(self, n_firms):
        """Setup simulation for given scale."""
        self.sim = bam.Simulation.init(
            n_firms=n_firms,
            n_households=n_firms * 5,
            n_banks=max(2, n_firms // 10),
            seed=42,
            logging={"default_level": "ERROR"},
        )

    def time_100_periods(self, n_firms):
        """Track how 100-period runtime scales with agent count."""
        self.sim.run(n_periods=100)

    def time_single_step(self, n_firms):
        """Track how single-step runtime scales."""
        self.sim.step()
