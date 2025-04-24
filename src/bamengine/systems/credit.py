import numpy as np
from ..components.firm_fin import FirmFin


def decide_credit_demand(fin: FirmFin) -> None:
    """Bᵢ = max(Wᵢ − Aᵢ, 0)."""
    fin.credit_demand[:] = np.maximum(fin.wage_bill - fin.net_worth, 0.0)
