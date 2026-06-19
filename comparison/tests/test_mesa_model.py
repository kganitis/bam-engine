import math

from comparison.orchestrator.params import canonical_params
from comparison.runners.mesa.model import BamModel


def _model(n_firms=100, n_households=500, n_banks=10, seed=0):
    return BamModel(n_firms, n_households, n_banks, canonical_params(), seed)


def test_initial_state_matches_reference():
    m = _model()
    assert len(m.firms) == 100 and len(m.households) == 500 and len(m.banks) == 10  # noqa: PT018
    f = next(iter(m.firms))
    assert f.price == 0.5 and f.production_prev == 2.5 and f.production == 0.0  # noqa: PT018
    assert f.labor_productivity == 0.5 and f.breakeven_price == 0.5  # noqa: PT018
    assert math.isclose(f.net_worth, 7.5) and math.isclose(f.total_funds, 7.5)  # noqa: PT018
    assert f.current_labor == 0 and f.n_vacancies == 0  # noqa: PT018
    assert math.isclose(f.wage_offer, 0.5 / 3)
    h = next(iter(m.households))
    assert h.employer == -1 and h.savings == 1.0 and h.wage == 0.0  # noqa: PT018
    b = next(iter(m.banks))
    assert b.equity_base == 5.0 and b.credit_supply == 0.0  # noqa: PT018
    assert m.avg_mkt_price == 0.5
    assert math.isclose(m.min_wage, (0.5 / 3) * 0.5)
    assert m.avg_mkt_price_history == [0.5] and m.inflation_history == [0.0]  # noqa: PT018
