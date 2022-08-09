"""
Unit and regression test for the automembrane package.
"""

# Import package, test suite, and other packages as needed
import sys

import automembrane
import automembrane.util as u
import jax
import numpy as np
import pytest
from automembrane.energy import ClosedPlaneCurveMaterial, OpenPlaneCurveMaterial
from jax.config import config

config.update("jax_enable_x64", True)


def test_automembrane_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "automembrane" in sys.modules


def test_closedplanecurvematerial():
    """Test ClosedPlaneCurveMaterial"""
    with np.testing.assert_no_gc_cycles():
        m = ClosedPlaneCurveMaterial()
        g, _ = u.ellipse()

        e = m.energy(g)
        _e = m._energy(g)

        assert e.shape == (3,)
        np.testing.assert_allclose(e, _e)
        assert np.all(e)

        f = m.force(g)
        _f = m._force(g)
        assert f.shape == (3, *g.shape)
        np.testing.assert_allclose(f, _f)

        local_f = jax.jacrev(m._energy)(g)
        np.testing.assert_allclose(f, -local_f)

        e_ef, f_ef = m.energy_force(g)
        np.testing.assert_allclose(e, e_ef)
        np.testing.assert_allclose(f, f_ef)

        m = ClosedPlaneCurveMaterial(0, 0, 0)
        e, f = m.energy_force(g)
        np.testing.assert_allclose(e, np.zeros_like(e))
        np.testing.assert_allclose(f, np.zeros_like(f))

        # Ensure last point is valid
        g2 = g[:-1]
        with pytest.raises(
            RuntimeError, match=r".*points are expected to be the same.*"
        ):
            m.energy(g2)
