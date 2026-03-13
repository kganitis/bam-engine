"""Tests for calibration.cost module."""

from __future__ import annotations

from unittest.mock import patch

from calibration.cost import SwapResult, classify_cost, parse_swaps, run_cost_analysis


class TestClassifyCost:
    """Tests for classify_cost."""

    def test_free(self):
        assert classify_cost(0.001) == "FREE"

    def test_cheap(self):
        assert classify_cost(0.003) == "CHEAP"

    def test_moderate(self):
        assert classify_cost(0.007) == "MODERATE"

    def test_expensive(self):
        assert classify_cost(0.015) == "EXPENSIVE"

    def test_zero_is_free(self):
        assert classify_cost(0.0) == "FREE"

    def test_boundary_cheap(self):
        assert classify_cost(0.002) == "CHEAP"

    def test_boundary_moderate(self):
        assert classify_cost(0.005) == "MODERATE"

    def test_boundary_expensive(self):
        assert classify_cost(0.010) == "EXPENSIVE"


class TestParseSwaps:
    """Tests for parse_swaps."""

    def test_single_param_multiple_values(self):
        swaps = parse_swaps(["beta=5.0,2.5"])
        assert swaps == {"beta": [5.0, 2.5]}

    def test_multiple_params(self):
        swaps = parse_swaps(["beta=5.0,2.5", "price_init=2.0,1.0"])
        assert swaps == {"beta": [5.0, 2.5], "price_init": [2.0, 1.0]}

    def test_integer_values(self):
        swaps = parse_swaps(["max_M=2,4"])
        assert swaps == {"max_M": [2, 4]}

    def test_string_values(self):
        swaps = parse_swaps(["job_search_method=vacancies_only,all_firms"])
        assert swaps == {"job_search_method": ["vacancies_only", "all_firms"]}


class TestSwapResult:
    """Tests for SwapResult dataclass."""

    def test_creation(self):
        sr = SwapResult(
            param="beta",
            value=5.0,
            base_combined=0.85,
            swap_combined=0.848,
            delta=-0.002,
            classification="CHEAP",
            pass_rate=1.0,
        )
        assert sr.classification == "CHEAP"
        assert sr.delta == -0.002


class TestRunCostAnalysis:
    """Tests for run_cost_analysis with mocked evaluation."""

    @patch("calibration.cost._evaluate_single_seed")
    def test_computes_swap_results(self, mock_eval):
        # Base: score=0.850, n_fail=0 for all seeds
        # Swap beta=2.5: score=0.849, n_fail=0 -- tiny delta (FREE)
        def side_effect(params, scenario, seed, n_periods):
            if params.get("beta") == 5.0:
                return params, scenario, 0.850, 0
            else:
                return params, scenario, 0.849, 0

        mock_eval.side_effect = side_effect

        results = run_cost_analysis(
            base_params={"beta": 5.0, "max_M": 4},
            swaps={"beta": [2.5]},
            scenario="baseline",
            n_seeds=3,
            n_workers=1,
        )
        assert len(results) == 1
        assert results[0].param == "beta"
        assert results[0].value == 2.5
        # delta ≈ 0.001 (< 0.002) -> FREE
        assert results[0].classification == "FREE"

    @patch("calibration.cost._evaluate_single_seed")
    def test_pass_rate_uses_n_fail(self, mock_eval):
        """Pass rate should be based on n_fail == 0, not score > 0."""
        # 2 seeds: one passes (n_fail=0), one fails (n_fail=1)
        calls = iter([(0.85, 0), (0.80, 1)])

        def side_effect(params, scenario, seed, n_periods):
            score, nf = next(calls)
            return params, scenario, score, nf

        mock_eval.side_effect = side_effect

        results = run_cost_analysis(
            base_params={"beta": 5.0},
            swaps={"beta": [2.5]},
            scenario="baseline",
            n_seeds=2,
            n_workers=1,
            base_combined=0.85,
        )
        assert results[0].pass_rate == 0.5  # 1 of 2 seeds passed
