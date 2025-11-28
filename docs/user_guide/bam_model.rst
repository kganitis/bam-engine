The BAM Model
=============

.. note::

   This section is under construction.

The BAM (Bottom-Up Adaptive Macroeconomics) model is an agent-based macroeconomic
model from the CATS (Complex Adaptive Trivial Systems) family, originally described
in Delli Gatti et al. (2011).

Agent Types
-----------

The model simulates three types of agents:

- **Firms**: Produce goods, hire workers, borrow from banks
- **Households**: Work for firms, consume goods, save income
- **Banks**: Provide credit to firms, collect interest

Markets
-------

Agents interact in three markets:

- **Labor Market**: Workers seek jobs, firms post vacancies and hire
- **Credit Market**: Firms request loans, banks supply credit
- **Goods Market**: Households purchase consumption goods from firms

Event Sequence
--------------

Each simulation period executes events in a specific order:

1. Planning: Firms decide production targets and prices
2. Labor Market: Wage setting, job applications, hiring
3. Credit Market: Credit supply/demand, loan applications
4. Production: Wage payments, goods production
5. Goods Market: Consumption decisions, shopping
6. Revenue: Firms collect revenue, pay dividends
7. Bankruptcy: Insolvent agents exit and are replaced

Topics to be covered:

* Detailed agent behaviors
* Market matching mechanisms
* Price and wage dynamics
* Bankruptcy and entry
* Calibration parameters
* References to the original literature
