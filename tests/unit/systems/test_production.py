# # tests/unit/systems/test_production.py
# """
# Unit tests for production systems.
# """
# from __future__ import annotations
#
# import numpy as np
#
# from bamengine.components import Employer, Worker
# from bamengine.systems.production import (
#     firms_pay_wages,
#     firms_run_production,
#     workers_receive_wage,
#     workers_update_contracts,
# )
# from tests.helpers.factories import (
#     mock_consumer,
#     mock_employer,
#     mock_producer,
#     mock_worker,
# )
#
#
# # ------------------------------------------------------------------ #
# #  firms_pay_wages                                                   #
# # ------------------------------------------------------------------ #
# def test_firms_pay_wages_debits_cash() -> None:
#     emp = mock_employer(
#         n=2,
#         current_labor=np.array([4, 0]),
#         wage_offer=np.array([1.0, 2.0]),
#         wage_bill=np.array([4.0, 0.0]),
#         total_funds=np.array([20.0, 7.0]),
#     )
#     before = emp.total_funds.copy()
#
#     firms_pay_wages(emp)
#
#     np.testing.assert_allclose(emp.total_funds, before - emp.wage_bill, rtol=1e-12)
#
#
# # ------------------------------------------------------------------ #
# #  consumers_receive_wage                                            #
# # ------------------------------------------------------------------ #
# def test_consumers_receive_wage_credits_income() -> None:
#     wrk = mock_worker(
#         n=2,
#         employed=np.array([1, 0], dtype=np.bool_),
#         wage=np.array([4.0, 3.0]),
#     )
#     con = mock_consumer(n=2, income=np.array([1.0, 5.0]))
#
#     workers_receive_wage(con, wrk)
#
#     # only consumer-0 is paid (worker-0 employed)
#     np.testing.assert_allclose(con.income, np.array([5.0, 5.0]))
#
#
# # ------------------------------------------------------------------ #
# #  firms_run_production                                              #
# # ------------------------------------------------------------------ #
# def test_firms_run_production_updates_output_and_stock() -> None:
#     emp = mock_employer(
#         n=2,
#         current_labor=np.array([4, 0]),
#     )
#     prod = mock_producer(
#         n=2,
#         labor_productivity=np.array([2.0, 3.0]),
#         production=np.zeros(2),
#         inventory=np.zeros(2),
#     )
#
#     firms_run_production(prod, emp)
#
#     expected = prod.labor_productivity * emp.current_labor
#     np.testing.assert_allclose(prod.production, expected)
#     np.testing.assert_allclose(prod.inventory, expected)
#
#
# # --------------------------------------------------------------------------- #
# #  deterministic micro-scenario helper                                        #
# # --------------------------------------------------------------------------- #
# def _mini_state() -> tuple[Employer, Worker]:
#     emp = mock_employer(
#         n=2,
#         current_labor=np.array([2, 1]),
#     )
#     wrk = mock_worker(
#         n=3,
#         employed=np.array([1, 1, 1], dtype=np.bool_),
#         employer=np.array([0, 0, 1]),
#         periods_left=np.array([2, 1, 1]),
#         wage=np.array([1.0, 1.0, 1.5]),
#     )
#     return emp, wrk
#
#
# # ------------------------------------------------------------------ #
# # 1. happy path – some contracts expire                              #
# # ------------------------------------------------------------------ #
# def test_contracts_expire_and_update_everything() -> None:
#     emp, wrk = _mini_state()
#
#     workers_update_contracts(wrk, emp)
#
#     # worker-side ----------------------------------------------------
#     assert wrk.employed.tolist() == [1, 0, 0]  # workers 1 & 2 expired
#     assert wrk.contract_expired.tolist() == [0, 1, 1]
#     assert wrk.employer_prev.tolist() == [-1, 0, 1]
#     assert wrk.employer.tolist() == [0, -1, -1]
#     assert wrk.wage.tolist() == [1.0, 0.0, 0.0]
#     assert wrk.periods_left.tolist() == [1, 0, 0]
#
#     # firm-side ------------------------------------------------------
#     # two heads left firm-0 and one head left firm-1
#     assert emp.current_labor.tolist() == [0, 0]
#
#
# # ------------------------------------------------------------------ #
# # 2. nothing expires – no-op branch                                  #
# # ------------------------------------------------------------------ #
# def test_contracts_no_expiration_no_change() -> None:
#     emp, wrk = _mini_state()
#     wrk.periods_left[:] = [5, 4, 3]  # far from expiry
#
#     before_emp = emp.current_labor.copy()
#     before_wrk = wrk.__dict__.copy()  # shallow ok – we only compare scalars
#
#     workers_update_contracts(wrk, emp)
#
#     assert np.array_equal(emp.current_labor, before_emp)
#     for k, v in before_wrk.items():
#         if isinstance(v, np.ndarray):
#             assert np.array_equal(v, getattr(wrk, k))
#
#
# # ------------------------------------------------------------------ #
# # 3. edge-case: periods_left already 0 (should be treated as expired)#
# # ------------------------------------------------------------------ #
# def test_contracts_already_zero_are_handled() -> None:
#     emp = mock_employer(
#         n=1,
#         current_labor=np.array([1]),
#     )
#     wrk = mock_worker(
#         n=1,
#         employed=np.array([1]),
#         employer=np.array([0]),
#         periods_left=np.array([0]),
#         wage=np.array([2.0]),
#     )
#
#     workers_update_contracts(wrk, emp)
#
#     assert wrk.employed[0] == 0
#     assert emp.current_labor[0] == 0
