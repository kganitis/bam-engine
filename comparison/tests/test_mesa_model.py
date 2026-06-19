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
    assert h.employer is None and h.savings == 1.0 and h.wage == 0.0  # noqa: PT018
    b = next(iter(m.banks))
    assert b.equity_base == 5.0 and b.credit_supply == 0.0  # noqa: PT018
    assert m.avg_mkt_price == 0.5
    assert math.isclose(m.min_wage, (0.5 / 3) * 0.5)
    assert m.avg_mkt_price_history == [0.5] and m.inflation_history == [0.0]  # noqa: PT018


def test_rng_is_deterministic():
    m1 = BamModel(3, 5, 1, canonical_params(), seed=123)
    m2 = BamModel(3, 5, 1, canonical_params(), seed=123)
    seq1 = [m1.random.random() for _ in range(5)]
    seq2 = [m2.random.random() for _ in range(5)]
    assert seq1 == seq2


def test_desired_labor_ceil_ratchet():
    m = _model(n_firms=1, n_households=5, n_banks=1)
    f = next(iter(m.firms))
    f.desired_production = 4.5
    f.labor_productivity = 0.5  # 4.5/0.5 = 9.0 -> 9
    f.decide_desired_labor()
    assert f.desired_labor == 9
    f.desired_production = 4.4  # 8.8 -> ceil 9 (ratchet)
    f.decide_desired_labor()
    assert f.desired_labor == 9


def test_plan_price_raise_floored_by_breakeven():
    m = _model(n_firms=1, n_households=5, n_banks=1, seed=1)
    f = next(iter(m.firms))
    m.avg_mkt_price = 1.0
    f.price = 0.5
    f.inventory = 0.0
    f.breakeven_price = 0.6  # sold out, underpriced -> raise, floored
    f.plan_price()
    assert f.price >= 0.6


def test_fire_excess_reduces_labor():
    m = _model(n_firms=1, n_households=5, n_banks=1, seed=2)
    f = next(iter(m.firms))
    hs = list(m.households)[:3]
    for h in hs:
        h.employer = f
        f.employees.add(h)
        f.current_labor += 1
    f.desired_labor = 1
    f.fire_excess_workers()
    assert f.current_labor == 1 and len(f.employees) == 1  # noqa: PT018
    fired = [h for h in hs if h.employer is None]
    assert len(fired) == 2 and all(h.fired and h.employer_prev is f for h in fired)  # noqa: PT018


def test_labor_round_hires_up_to_vacancies_with_contract_theta():
    m = _model(n_firms=2, n_households=5, n_banks=1, seed=3)
    from comparison.runners.mesa.markets import run_labor_market

    firms = list(m.firms)
    a, b = firms
    a.n_vacancies = 1
    a.wage_offer = 1.0
    b.n_vacancies = 1
    b.wage_offer = 0.5
    for h in m.households:  # all unemployed apply to both, A ranked first (higher wage)
        h.job_apps = [a, b]
    run_labor_market(m)
    hired_a = [h for h in m.households if h.employer is a]
    hired_b = [h for h in m.households if h.employer is b]
    assert len(hired_a) == 1 and len(hired_b) == 1  # noqa: PT018
    assert all(h.periods_left == m.p["theta"] for h in hired_a + hired_b)
    assert a.current_labor == 1 and b.current_labor == 1  # noqa: PT018


def test_wage_bill_sums_employee_wages():
    m = _model(n_firms=1, n_households=3, n_banks=1)
    f = next(iter(m.firms))
    for h, w in zip(list(m.households), [1.0, 2.0, 3.0], strict=True):
        h.employer = f
        h.wage = w
        f.employees.add(h)
    f.calc_wage_bill()
    assert f.wage_bill == 6.0
