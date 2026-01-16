"""
Credit market events for credit supply, demand, and loan provision.

This module defines the credit market phase events that execute after labor
market events. Banks decide credit supply and interest rates, firms determine
credit needs, and loans are matched through a sequential application process.
Firms that fail to secure sufficient credit fire workers to match available funds.

Event Sequence
--------------
The credit market events execute in this order:

1. BanksDecideCreditSupply - Banks set total lendable funds based on equity
2. BanksDecideInterestRate - Banks set interest rates with random markup
3. FirmsDecideCreditDemand - Firms calculate funding shortfall
4. FirmsCalcFinancialFragility - Firms calculate leverage and fragility
5. FirmsPrepareLoanApplications - Firms select banks to apply to (sorted by rate)
6. FirmsSendOneLoanApp ↔ BanksProvideLoans - Interleaved application/provision rounds (max_H times)
7. FirmsFireWorkers - Firms with insufficient funds lay off workers

The send/provide rounds are interleaved to simulate sequential matching: firms
send one application, banks evaluate and provide loans, then the next round
begins. This repeats max_H times to process all applications.

Design Notes
------------
- Events operate on borrower, lender, and loanbook (Borrower, Lender, LoanBook)
- Banks rank loan applicants by net worth (descending) for default risk assessment
- Firms fire most expensive workers first to minimize layoffs
- Credit supply constrained by bank equity and capital requirement (v parameter)
- Interest rates: :math:`r = \\bar{r} \\times (1 + \\varepsilon)`, where :math:`\\varepsilon \\sim U(0, h_\\phi)`

Examples
--------
Execute credit market events:

>>> import bamengine as be
>>> sim = be.Simulation.init(n_firms=100, n_banks=10, seed=42)
>>> # Credit market events run as part of default pipeline
>>> sim.step()

Execute individual credit market event:

>>> event = sim.get_event("banks_decide_credit_supply")
>>> event.execute(sim)
>>> sim.lend.credit_supply.mean()  # doctest: +SKIP
2500.0

Check loan book after credit provision:

>>> sim.lb.size  # doctest: +SKIP
45
>>> sim.lb.principal[: sim.lb.size].sum()  # doctest: +SKIP
1250.0

See Also
--------
bamengine.events._internal.credit_market : System function implementations
Borrower : Firm credit demand and financial state
Lender : Bank credit supply and interest rates
LoanBook : Loan relationship between borrowers and lenders
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.simulation import Simulation


@event
class BanksDecideCreditSupply:
    """
    Banks decide total credit supply based on equity and capital requirement.

    Banks set their maximum lendable funds based on their equity base and the
    regulatory capital requirement coefficient (v). Lower v means banks can
    lend more relative to their equity (higher leverage).

    Algorithm
    ---------
    For each bank k:

    .. math::
        C_k = E_k / v

    where:

    - :math:`C_k`: total credit supply (lendable funds) for bank k
    - :math:`E_k`: equity base of bank k
    - :math:`v`: capital requirement coefficient (Simulation parameter)

    Mathematical Notation
    ---------------------
    .. math::
        C_k = \\frac{E_k}{v}

    Typical value: v = 0.1 implies banks can lend 10× their equity base.

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_banks=10, seed=42)
    >>> event = sim.get_event("banks_decide_credit_supply")
    >>> event.execute(sim)

    Check credit supply:

    >>> sim.lend.credit_supply.mean()  # doctest: +SKIP
    2500.0

    Verify credit supply formula:

    >>> import numpy as np
    >>> expected_supply = sim.lend.equity_base / sim.v
    >>> np.allclose(sim.lend.credit_supply, expected_supply)
    True

    Check total available credit:

    >>> total_credit = sim.lend.credit_supply.sum()
    >>> total_credit  # doctest: +SKIP
    25000.0

    Notes
    -----
    This event must execute at the start of the credit market phase, before
    BanksDecideInterestRate and FirmsDecideCreditDemand.

    The capital requirement coefficient v is a Simulation-level parameter
    (not in config), accessed via `sim.v`.

    Credit supply is reset each period based on current equity. Any unused
    credit from previous periods does not carry over.

    See Also
    --------
    BanksProvideLoans : Uses credit_supply to provision loans
    Lender : Bank state with equity_base and credit_supply
    bamengine.events._internal.credit_market.banks_decide_credit_supply : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.credit_market import banks_decide_credit_supply

        banks_decide_credit_supply(sim.lend, v=sim.v)


@event
class BanksDecideInterestRate:
    """
    Banks set interest rates as markup over base rate with random shock.

    Banks apply a random markup to the baseline policy rate to set their lending
    rates. The markup introduces heterogeneity in bank rates and competition for
    low-rate lenders.

    Algorithm
    ---------
    For each bank k:

    1. Generate rate shock: :math:`\\varepsilon_k \\sim U(0, h_\\phi)`
    2. Apply markup: :math:`r_k = \\bar{r} \\times (1 + \\varepsilon_k)`

    Mathematical Notation
    ---------------------
    .. math::
        r_k = \\bar{r} \\times (1 + \\varepsilon_k)

    where:

    - :math:`r_k`: interest rate charged by bank k
    - :math:`\\bar{r}`: baseline policy rate (Simulation parameter)
    - :math:`\\varepsilon_k`: random shock :math:`\\sim U(0, h_\\phi)`
    - :math:`h_\\phi`: maximum interest rate shock parameter (config)

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_banks=10, seed=42)
    >>> event = sim.get_event("banks_decide_interest_rate")
    >>> event.execute(sim)

    Check interest rates:

    >>> sim.lend.interest_rate.mean()  # doctest: +SKIP
    0.035

    Verify rates are above base rate:

    >>> (sim.lend.interest_rate >= sim.r_bar).all()
    True

    Find lowest-rate bank:

    >>> import numpy as np
    >>> cheapest_bank = np.argmin(sim.lend.interest_rate)
    >>> sim.lend.interest_rate[cheapest_bank]  # doctest: +SKIP
    0.031

    Notes
    -----
    This event must execute after BanksDecideCreditSupply and before
    FirmsPrepareLoanApplications (firms sort banks by rate).

    The baseline policy rate :math:`\\bar{r}` is a Simulation-level parameter accessed via `sim.r_bar`.

    All banks charge rates :math:`\\geq \\bar{r}` since shock :math:`\\varepsilon \\geq 0`.

    See Also
    --------
    FirmsPrepareLoanApplications : Firms sort banks by interest rate
    BanksProvideLoans : Uses interest_rate for new loans
    bamengine.events._internal.credit_market.banks_decide_interest_rate : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.credit_market import banks_decide_interest_rate

        banks_decide_interest_rate(
            sim.lend,
            r_bar=sim.r_bar,
            h_phi=sim.config.h_phi,
            rng=sim.rng,
        )


@event
class FirmsDecideCreditDemand:
    """
    Firms calculate credit demand based on funding shortfall.

    Firms need to pay their wage bill but may lack sufficient funds (net worth).
    Credit demand is the shortfall between wage obligations and available funds.

    Algorithm
    ---------
    For each firm i:

    .. math::
        B_i = \\max(W_i - A_i, 0)

    where:

    - :math:`B_i`: credit demand (amount firm needs to borrow)
    - :math:`W_i`: wage bill (total wages owed to workers)
    - :math:`A_i`: net worth (firm's current funds/assets)

    Firms with :math:`A_i \\geq W_i` have zero credit demand (self-financed).

    Mathematical Notation
    ---------------------
    .. math::
        B_i = \\max(0, W_i - A_i)

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("firms_decide_credit_demand")
    >>> event.execute(sim)

    Check total credit demand:

    >>> sim.bor.credit_demand.sum()  # doctest: +SKIP
    1250.0

    Find firms needing credit:

    >>> import numpy as np
    >>> needs_credit = sim.bor.credit_demand > 0
    >>> needs_credit.sum()  # doctest: +SKIP
    45

    Verify credit demand formula:

    >>> shortfall = np.maximum(sim.bor.wage_bill - sim.bor.net_worth, 0)
    >>> np.allclose(sim.bor.credit_demand, shortfall)
    True

    Notes
    -----
    This event must execute after FirmsCalcWageBill (need wage_bill) and before
    FirmsPrepareLoanApplications.

    Firms with negative net worth may have very high credit demand (potentially
    exceeding available credit supply).

    Credit demand is zero for self-financed firms (net_worth >= wage_bill).

    See Also
    --------
    FirmsCalcWageBill : Calculates wage_bill used in credit demand
    FirmsCalcFinancialFragility : Uses credit_demand to calculate leverage
    FirmsPrepareLoanApplications : Firms with credit_demand > 0 apply for loans
    bamengine.events._internal.credit_market.firms_decide_credit_demand : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.credit_market import firms_decide_credit_demand

        firms_decide_credit_demand(sim.bor)


@event
class FirmsCalcFinancialFragility:
    """
    Firms calculate projected financial fragility metric for credit evaluation.

    The fragility metric is the leverage ratio (debt-to-equity). Higher fragility
    indicates greater default risk. Banks may use this metric (implicitly via
    net worth ranking) to assess creditworthiness.

    Algorithm
    ---------
    For each firm i:

    1. Calculate leverage: :math:`l_i = B_i / A_i` (if :math:`A_i > 0`, else :math:`l_i = 0`)
    2. Cap leverage at :math:`B_i` to prevent explosion for small :math:`A_i`

    Mathematical Notation
    ---------------------
    .. math::
        f_i = \\frac{B_i}{A_i}

    where:

    - :math:`f_i`: projected financial fragility (leverage)
    - :math:`B_i`: credit demand
    - :math:`A_i`: net worth

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> # First calculate credit demand
    >>> sim.get_event("firms_decide_credit_demand")().execute(sim)
    >>> # Then calculate metrics
    >>> event = sim.get_event("firms_calc_financial_fragility")
    >>> event.execute(sim)

    Check fragility distribution:

    >>> sim.bor.projected_fragility.mean()  # doctest: +SKIP
    0.15

    Find high-fragility firms:

    >>> import numpy as np
    >>> high_risk = sim.bor.projected_fragility > 0.5
    >>> high_risk.sum()  # doctest: +SKIP
    8

    Verify fragility calculation:

    >>> # For firms with positive net worth
    >>> pos_net_worth = sim.bor.net_worth > 0
    >>> leverage = (
    ...     sim.bor.credit_demand[pos_net_worth] / sim.bor.net_worth[pos_net_worth]
    ... )
    >>> # Fragility is capped at credit_demand
    >>> expected_fragility = np.minimum(leverage, sim.bor.credit_demand[pos_net_worth])
    >>> actual_fragility = sim.bor.projected_fragility[pos_net_worth]
    >>> np.allclose(actual_fragility, expected_fragility)
    True

    Notes
    -----
    This event must execute after FirmsDecideCreditDemand and before
    FirmsPrepareLoanApplications.

    Fragility is calculated but not directly used by banks in the current
    implementation. Banks rank applicants by net worth instead. Future
    extensions could incorporate fragility into credit decisions.

    Firms with zero or negative net worth have undefined leverage. The
    implementation sets leverage = 0 for these cases to avoid division errors.

    See Also
    --------
    FirmsDecideCreditDemand : Calculates credit_demand used in leverage
    BanksProvideLoans : Banks evaluate creditworthiness (by net worth)
    Borrower : Financial state with credit_demand, net_worth
    bamengine.events._internal.credit_market.firms_calc_financial_fragility : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.credit_market import (
            firms_calc_financial_fragility,
        )

        firms_calc_financial_fragility(
            sim.bor,
            fragility_cap_method=sim.config.fragility_cap_method,
        )


@event
class FirmsPrepareLoanApplications:
    """
    Firms select banks to apply to, sorted by interest rate (ascending).

    Firms with positive credit demand build a loan application queue by sampling
    banks and sorting them by interest rate. Firms prefer lower-rate banks to
    minimize borrowing costs.

    Algorithm
    ---------
    For each firm i with :math:`B_i > 0` (credit demand):

    1. Sample min(max_H, n_banks) banks randomly
    2. Sort sampled banks by interest rate (ascending)
    3. Store sorted application queue in firm's buffer

    Mathematical Notation
    ---------------------
    For firm i with :math:`B_i > 0`:

    .. math::
        \\text{Sample}_i \\sim \\text{Random}(\\{1, ..., K\\}, k=\\min(H, K), \\text{replace}=False)

    Then sort by rate:

    .. math::
        \\text{Queue}_i = \\text{argsort}_{\\text{asc}}(r_k \\text{ for } k \\in \\text{Sample}_i)

    where :math:`K` = n_banks, :math:`H` = max_H.

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_banks=10, seed=42)
    >>> # First calculate credit demand
    >>> sim.get_event("firms_decide_credit_demand")().execute(sim)
    >>> # Then prepare applications
    >>> event = sim.get_event("firms_prepare_loan_applications")
    >>> event.execute(sim)

    Check firms with credit demand:

    >>> import numpy as np
    >>> needs_credit = sim.bor.credit_demand > 0
    >>> needs_credit.sum()  # doctest: +SKIP
    45

    Inspect application queue for a firm:

    >>> firm_ids = np.where(needs_credit)[0]
    >>> if len(firm_ids) > 0:
    ...     firm_id = firm_ids[0]
    ...     targets = sim.bor.loan_apps_targets[firm_id]
    ...     # First 3 bank targets
    ...     targets[:3]  # doctest: +SKIP
    array([2, 7, 1])

    Verify banks are sorted by rate:

    >>> if len(firm_ids) > 0:
    ...     firm_id = firm_ids[0]
    ...     bank_ids = sim.bor.loan_apps_targets[firm_id, : sim.config.max_H]
    ...     bank_ids = bank_ids[bank_ids >= 0]  # Exclude -1 (padding)
    ...     rates = sim.lend.interest_rate[bank_ids]
    ...     # Check rates are non-decreasing
    ...     np.all(rates[:-1] <= rates[1:])  # doctest: +SKIP
    True

    Notes
    -----
    This event must execute after BanksDecideInterestRate and FirmsDecideCreditDemand.

    Only firms with positive credit demand prepare applications. Self-financed
    firms (:math:`B_i = 0`) are skipped.

    Firms sample banks randomly then sort by rate. This means firms may miss
    the absolute lowest-rate bank if it's not in their random sample.

    See Also
    --------
    BanksDecideInterestRate : Sets rates used for sorting
    FirmsSendOneLoanApp : Processes applications from queue
    Borrower : Financial state with loan application queue
    bamengine.events._internal.credit_market.firms_prepare_loan_applications : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.credit_market import (
            firms_prepare_loan_applications,
        )

        firms_prepare_loan_applications(
            sim.bor,
            sim.lend,
            max_H=sim.config.max_H,
            rng=sim.rng,
        )


@event
class FirmsSendOneLoanApp:
    """
    Firms send one round of loan applications to banks.

    Each firm with credit demand sends one application from their queue to the
    corresponding bank. This event is repeated max_H times in the pipeline,
    interleaved with BanksProvideLoans, to process all applications sequentially.

    Algorithm
    ---------
    For each firm i with credit demand (:math:`B_i > 0`):

    1. Check if firm has applications remaining in queue (head pointer >= 0)
    2. Pop next target bank from firm's application queue
    3. Append firm ID to bank's application queue
    4. Advance firm's queue pointer

    Examples
    --------
    Execute this event (typically in loop with BanksProvideLoans):

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_banks=10, seed=42)
    >>> # First prepare applications
    >>> sim.get_event("firms_prepare_loan_applications")().execute(sim)
    >>> # Process one round
    >>> event = sim.get_event("firms_send_one_loan_app")
    >>> event.execute(sim)

    Process all application rounds:

    >>> max_H = sim.config.max_H
    >>> for _ in range(max_H):
    ...     sim.get_event("firms_send_one_loan_app")().execute(sim)
    ...     sim.get_event("banks_provide_loans")().execute(sim)

    Notes
    -----
    This event must execute after FirmsPrepareLoanApplications (need application queues).

    This event is typically repeated max_H times, interleaved with BanksProvideLoans,
    to simulate sequential matching.

    See Also
    --------
    FirmsPrepareLoanApplications : Prepares application queues
    BanksProvideLoans : Processes applications and provides loans
    bamengine.events._internal.credit_market.firms_send_one_loan_app : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.credit_market import firms_send_one_loan_app

        firms_send_one_loan_app(sim.bor, sim.lend, rng=sim.rng)


@event
class BanksProvideLoans:
    """
    Banks process loan applications and provide credit based on net worth ranking.

    Each bank evaluates its application queue, ranking applicants by net worth
    (descending) to assess default risk. Banks provide loans up to available
    credit supply and record them in the LoanBook. This event is repeated max_H
    times, interleaved with FirmsSendOneLoanApp.

    Algorithm
    ---------
    For each bank k:

    1. Pop one batch of applications from bank's application queue
    2. Sort applicants by net worth (descending - prefer safer borrowers)
    3. For each applicant i in sorted order:
       - Check if bank has credit supply remaining (:math:`C_k > 0`)
       - Check if firm still needs credit (:math:`B_i > 0`)
       - Grant loan amount: :math:`L = \\min(C_k, B_i)`
       - Update firm's total_funds: :math:`A_i \\leftarrow A_i + L`
       - Update firm's credit_demand: :math:`B_i \\leftarrow B_i - L`
       - Update bank's credit_supply: :math:`C_k \\leftarrow C_k - L`
       - Add loan to LoanBook with principal=L, rate=:math:`r_k`
    4. Clear bank's application queue if credit supply exhausted

    Mathematical Notation
    ---------------------
    For bank k processing application from firm i:

    .. math::
        L = \\min(C_k, B_i)

        A_i \\leftarrow A_i + L

        B_i \\leftarrow B_i - L

        C_k \\leftarrow C_k - L

    Loan recorded as: (borrower=i, lender=k, principal=L, rate=r_k, interest=L×r_k, debt=L×(1+r_k))

    Examples
    --------
    Execute this event (typically in loop with FirmsSendOneLoanApp):

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_banks=10, seed=42)
    >>> # Prepare and send applications
    >>> sim.get_event("firms_prepare_loan_applications")().execute(sim)
    >>> sim.get_event("firms_send_one_loan_app")().execute(sim)
    >>> # Process loans
    >>> event = sim.get_event("banks_provide_loans")
    >>> event.execute(sim)

    Check loan book after provision:

    >>> sim.lb.size  # doctest: +SKIP
    15
    >>> sim.lb.principal[: sim.lb.size].sum()  # doctest: +SKIP
    425.0

    Verify loan book arithmetic:

    >>> import numpy as np
    >>> loans = sim.lb
    >>> p = loans.principal[: loans.size]
    >>> r = loans.rate[: loans.size]
    >>> interest = loans.interest[: loans.size]
    >>> debt = loans.debt[: loans.size]
    >>> np.allclose(interest, p * r)
    True
    >>> np.allclose(debt, p * (1 + r))
    True

    Notes
    -----
    This event must execute after FirmsSendOneLoanApp (need applications to process).

    This event is typically repeated max_H times, interleaved with FirmsSendOneLoanApp,
    to simulate sequential matching.

    Banks rank applicants by net worth (descending) as a proxy for creditworthiness.
    Higher net worth implies lower default risk.

    Partial loan fulfillment is possible: if firm requests 100 but bank only has
    50 credit supply, firm receives 50 (partial grant).

    See Also
    --------
    FirmsSendOneLoanApp : Sends applications to banks
    LoanBook : Stores loan relationships
    Lender : Bank state with credit_supply
    Borrower : Firm state with credit_demand, total_funds
    bamengine.events._internal.credit_market.banks_provide_loans : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.credit_market import banks_provide_loans

        banks_provide_loans(
            sim.bor,
            sim.lb,
            sim.lend,
            r_bar=sim.r_bar,
            h_phi=sim.config.h_phi,
            loan_priority_method=sim.config.loan_priority_method,
            max_loan_to_net_worth=sim.config.max_loan_to_net_worth,
        )


@event
class FirmsFireWorkers:
    """
    Firms with insufficient funds after credit provision fire workers.

    Firms that failed to secure enough credit to cover their wage bill must lay
    off workers to match their available funds. By default, firms fire the most
    expensive workers first to minimize the number of layoffs.

    Algorithm
    ---------
    For each firm i with :math:`W_i > A_i` (wage bill exceeds available funds):

    1. Calculate unfunded amount: :math:`U_i = W_i - A_i`
    2. Get list of employees sorted by wage (descending)
    3. Fire workers from the list until unfunded amount <= 0:
       - Set worker's employer = -1 (unemployed)
       - Set worker's wage = 0
       - Set worker's fired flag = True
       - Decrement firm's current_labor
       - Reduce firm's wage_bill by worker's wage
       - Reduce unfunded amount: :math:`U_i \\leftarrow U_i - w_j`
    4. Update firm's wage_bill to match new labor force

    Mathematical Notation
    ---------------------
    For firm i with :math:`W_i > A_i`:

    .. math::
        U_i = W_i - A_i

    Fire workers j in descending wage order until:

    .. math::
        \\sum_{j \\in \\text{fired}} w_j \\geq U_i

    Update state:

    .. math::
        L_i \\leftarrow L_i - |\\text{fired}|

        W_i \\leftarrow \\sum_{j \\in \\text{remaining}} w_j

    Examples
    --------
    Execute this event:

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, n_households=500, seed=42)
    >>> # Simulate firms with insufficient credit
    >>> event = sim.get_event("firms_fire_workers")
    >>> event.execute(sim)

    Check fired workers:

    >>> import numpy as np
    >>> fired_mask = sim.wrk.fired
    >>> fired_mask.sum()  # doctest: +SKIP
    8

    Verify fired workers are unemployed:

    >>> (sim.wrk.employer[fired_mask] == -1).all()
    True

    Check firms that fired workers:

    >>> # Firms that fired should now have wage_bill <= total_funds
    >>> firms_that_fired = np.unique(sim.wrk.employer_prev[fired_mask])
    >>> firms_that_fired = firms_that_fired[firms_that_fired >= 0]
    >>> for firm_id in firms_that_fired:
    ...     assert sim.emp.wage_bill[firm_id] <= sim.bor.total_funds[firm_id]

    Notes
    -----
    This event must execute after all BanksProvideLoans rounds complete.

    Firing strategy: most expensive workers first minimizes the number of layoffs
    (better to fire one high-wage worker than multiple low-wage workers).

    Fired workers have their `fired` flag set to True, which affects their job
    search behavior in the next period (loyalty rule does not apply).

    The wage_bill is recalculated after firing to reflect the new labor force.

    See Also
    --------
    BanksProvideLoans : Provides credit to firms
    FirmsCalcWageBill : Recalculates wage_bill after firing
    Worker : Employment state with fired flag
    Employer : Labor force state with current_labor, wage_bill
    bamengine.events._internal.credit_market.firms_fire_workers : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.credit_market import firms_fire_workers

        firms_fire_workers(
            sim.emp,
            sim.wrk,
            method=sim.config.firing_method,
            rng=sim.rng,
        )
