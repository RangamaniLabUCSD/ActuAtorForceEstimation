"""
Unit and regression test for the testproject package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import automembrane


def test_automembrane_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "automembrane" in sys.modules

    