Statistical Tools
=================

The ``stats`` module provides pure statistical functions used by both internal
validity and sensitivity analysis.


HP Filter
---------

Hodrick-Prescott filter for trend-cycle decomposition. Solves the penalized
least-squares problem:

.. math::

   \min_{\tau} \sum_{t=1}^{T} (y_t - \tau_t)^2
   + \lambda \sum_{t=3}^{T} (\Delta^2 \tau_t)^2

via the sparse linear system :math:`(I + \lambda K^T K)\tau = y`, where
:math:`K` is the second-difference matrix. Uses
``scipy.sparse.linalg.spsolve`` for efficiency.

.. code-block:: python

   from validation.robustness.stats import hp_filter

   trend, cycle = hp_filter(series, lamb=1600.0)


Cross-Correlation
-----------------

Computes the cross-correlation function:

.. math::

   \rho(k) = \text{corr}(x_t, \, y_{t+k})

for integer lags :math:`k = -\text{max\_lag}, \ldots, +\text{max\_lag}`.
At lag :math:`k=0`, this is the contemporaneous correlation. Positive
:math:`k` means :math:`y` *leads* :math:`x`.

.. code-block:: python

   from validation.robustness.stats import cross_correlation

   corrs = cross_correlation(gdp_cycle, unemployment_cycle, max_lag=4)


AR Fitting
----------

Fits an autoregressive model via ordinary least squares:

.. math::

   y_t = c + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t

using ``np.linalg.lstsq``. Returns coefficients
:math:`[c, \phi_1, \ldots, \phi_p]` and :math:`R^2`.

.. code-block:: python

   from validation.robustness.stats import fit_ar

   coeffs, r_squared = fit_ar(gdp_cycle, order=2)


Impulse-Response Function
--------------------------

Simulates the response to a unit shock at :math:`t=0` through the AR
recursion. For a stable AR(1) with coefficient :math:`\phi`, the IRF is
:math:`\phi^t` (exponential decay).

.. code-block:: python

   from validation.robustness.stats import impulse_response

   irf = impulse_response(coeffs, n_periods=20)


API Reference
-------------

.. automodule:: validation.robustness.stats
   :members:
   :undoc-members:
