from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from bamengine import Simulation
import logging

log = logging.getLogger(__name__)


@dataclass
class DataCollector:
    gdp: list[float] = field(default_factory=list)
    unemployment_rate: list[float] = field(default_factory=list)
    inflation_rate: list[float] = field(default_factory=list)
    productivity_wage_ratio: list[float] = field(default_factory=list)

    def capture(self, sim: Simulation) -> None:
        # Real GDP
        current_real_gdp = float(sim.prod.production.sum())
        self.gdp.append(current_real_gdp)

        # Unemployment Rate
        current_unemployment = float(sim.ec.unemp_rate_history[-1])
        self.unemployment_rate.append(current_unemployment)

        # Annual Inflation Rate
        current_annual_inflation = float(sim.ec.inflation_history[-1])
        self.inflation_rate.append(current_annual_inflation)

        # Productivity to Real Wage Ratio
        avg_productivity = float(sim.prod.labor_productivity.mean())
        if sim.ec.avg_mkt_price > 0:
            employed_wages = sim.wrk.wage[sim.wrk.employed]
            avg_nominal_wage = (
                float(employed_wages.mean()) if len(employed_wages) > 0 else 0.0
            )
            avg_real_wage = avg_nominal_wage / sim.ec.avg_mkt_price
            ratio = avg_productivity / avg_real_wage if avg_real_wage > 0 else 0.0
        else:
            ratio = 0.0
        self.productivity_wage_ratio.append(ratio)

        log.debug(
            f"Real GDP={current_real_gdp:.2f}, "
            f"Unemployment={current_unemployment:.3f}, "
            f"Inflation={current_annual_inflation * 100:.2f}, "
            f"Prod / Real Wage={ratio:.2f}, "
        )

    def get_arrays(self) -> dict[str, NDArray[np.float64]]:
        return {
            "gdp": np.array(self.gdp),
            "unemployment_rate": np.array(self.unemployment_rate),
            "inflation_rate": np.array(self.inflation_rate),
            "productivity_wage_ratio": np.array(self.productivity_wage_ratio),
        }
