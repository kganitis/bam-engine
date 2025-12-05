"""
Role (Component) classes for BAM Engine agents.

This package defines the six role classes that represent different aspects
of agent behavior in the BAM model. Each role is a dataclass containing
NumPy arrays for agent state, enabling efficient vectorized operations.

Role Design
-----------
- **Roles are components**: Each role represents one aspect of agent behavior
- **NumPy arrays**: All fields are 1D or 2D NumPy arrays (index = agent ID)
- **Auto-registration**: All roles inherit from Role and auto-register
- **Shared state**: Some arrays are shared between roles (e.g., total_funds)
- **Scratch buffers**: Optional fields for temporary data (marked repr=False)

Agent-Role Mapping
------------------
In the BAM model, agents have multiple roles:

- **Firms** (n_firms): Producer + Employer + Borrower
- **Households** (n_households): Worker + Consumer
- **Banks** (n_banks): Lender

For example, firm 5 has state in Producer[5], Employer[5], and Borrower[5].

Available Roles
---------------
Producer : Role
    Production and pricing state for firms
Employer : Role
    Labor hiring and wage state for firms
Borrower : Role
    Financial and credit state for firms
Worker : Role
    Employment and wage state for households
Consumer : Role
    Consumption and savings state for households
Lender : Role
    Credit supply and interest rate state for banks

Examples
--------
Access roles from simulation:

>>> import bamengine as bam
>>> sim = bam.Simulation.init(n_firms=100, n_households=500, seed=42)
>>> prod = sim.prod  # Producer role (100 firms)
>>> wrk = sim.wrk  # Worker role (500 households)
>>> prod.price.shape
(100,)
>>> wrk.wage.shape
(500,)

Use role registry:

>>> from bamengine import get_role
>>> ProducerClass = get_role("Producer")
>>> import numpy as np
>>> prod = ProducerClass(
...     price=np.ones(10),
...     production=np.zeros(10),
...     inventory=np.zeros(10),
...     expected_demand=np.ones(10),
...     desired_production=np.ones(10),
...     labor_productivity=np.ones(10) * 2.0,
...     breakeven_price=np.ones(10),
... )
>>> prod.price
array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

See Also
--------
:class:`bamengine.core.Role` : Base class for all roles
:class:`bamengine.Simulation` : Simulation facade with role instances
"""

from bamengine.roles.borrower import Borrower
from bamengine.roles.consumer import Consumer
from bamengine.roles.employer import Employer
from bamengine.roles.lender import Lender
from bamengine.roles.producer import Producer
from bamengine.roles.worker import Worker

__all__ = [
    "Borrower",
    "Consumer",
    "Employer",
    "Lender",
    "Producer",
    "Worker",
]
