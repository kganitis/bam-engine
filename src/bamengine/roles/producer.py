"""
Producer role for firms.

Represents the production and pricing aspect of firm behavior in the BAM model.
Each firm (agent) produces consumer goods using labor and sells them at a price.
"""

from dataclasses import field

from bamengine.core.decorators import role
from bamengine.typing import Bool1D, Float1D


@role
class Producer:
    """
    Producer role for firms.

    Represents the production and pricing state for firms. Each array index
    corresponds to a firm ID (0 to n_firms-1).

    Parameters
    ----------
    production : Float1D
        Current period's production level (units of goods produced this period).
        Set by ``firms_run_production`` event, zeroed at start of planning phase.
    production_prev : Float1D
        Previous period's production level, used as planning signal.
        Set by ``firms_run_production`` alongside production, retained across
        the planning phase for use in ``firms_decide_desired_production``.
    inventory : Float1D
        Unsold goods from previous periods.
    expected_demand : Float1D
        Expected demand based on past sales history.
    desired_production : Float1D
        Target production level for next period (based on expected demand).
    labor_productivity : Float1D
        Units of output per worker.
    breakeven_price : Float1D
        Minimum price needed to cover wage costs.
    price : Float1D
        Current selling price for goods.
    prod_shock : Float1D, optional
        Scratch buffer for production shock calculations (not persisted).
    prod_mask_up : Bool1D, optional
        Scratch buffer for upward production adjustment mask (not persisted).
    prod_mask_dn : Bool1D, optional
        Scratch buffer for downward production adjustment mask (not persisted).
    price_shock : Float1D, optional
        Scratch buffer for price shock calculations (not persisted).

    Examples
    --------
    Access from simulation:

    >>> import bamengine as bam
    >>> sim = bam.Simulation.init(n_firms=100, seed=42)
    >>> prod = sim.prod
    >>> prod.price.shape
    (100,)
    >>> prod.production.mean()  # doctest: +SKIP
    52.5

    Check which firms have inventory:

    >>> import numpy as np
    >>> has_inventory = prod.inventory > 0
    >>> has_inventory.sum()  # doctest: +SKIP
    15

    Find firms with high productivity:

    >>> high_prod_mask = prod.labor_productivity > 2.0
    >>> high_prod_ids = np.where(high_prod_mask)[0]
    >>> high_prod_ids.shape  # doctest: +SKIP
    (45,)

    Notes
    -----
    The Producer role is one of three roles assigned to firms in the BAM model:

    - Producer: production and pricing
    - Employer: labor hiring and wages (see Employer)
    - Borrower: finance and credit (see Borrower)

    These roles share some arrays (e.g., wage_bill) for memory efficiency.

    See Also
    --------
    :class:`~bamengine.roles.Employer` : Labor hiring role for firms
    :class:`~bamengine.roles.Borrower` : Financial role for firms
    :mod:`bamengine.events._internal.planning` : Production planning logic
    :mod:`bamengine.events._internal.production` : Production execution logic
    """

    production: Float1D
    production_prev: Float1D
    inventory: Float1D
    expected_demand: Float1D
    desired_production: Float1D
    labor_productivity: Float1D
    breakeven_price: Float1D
    price: Float1D

    # Scratch buffers (optional for performance)
    prod_shock: Float1D | None = field(default=None, repr=False)
    prod_mask_up: Bool1D | None = field(default=None, repr=False)
    prod_mask_dn: Bool1D | None = field(default=None, repr=False)
    price_shock: Float1D | None = field(default=None, repr=False)
