"""Performance test configuration.

Disables coverage instrumentation for performance tests to ensure
accurate timing measurements. Coverage tracing (sys.settrace) adds
40-60% overhead that distorts benchmarks and regression baselines.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_coverage(request):
    """Suspend coverage measurement during performance tests."""
    cov_plugin = request.config.pluginmanager.get_plugin("_cov")
    if cov_plugin and cov_plugin.cov_controller:
        cov_plugin.cov_controller.cov.stop()
        yield
        cov_plugin.cov_controller.cov.start()
    else:
        yield
