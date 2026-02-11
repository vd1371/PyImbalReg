"""Shared pytest fixtures for PyImbalReg tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def random_state():
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def sample_df(random_state):
    """Small numeric DataFrame, last column is target (imbalanced)."""
    np.random.seed(random_state)
    n = 100
    # Target with clear extremes so relevance/threshold make sense
    y = np.concatenate([
        np.random.normal(0, 1, 80),
        np.random.normal(5, 0.5, 15),
        np.array([10.0, 11.0]),
    ])
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    return pd.DataFrame({"x1": X1, "x2": X2, "y": y})


@pytest.fixture
def sample_df_with_categorical(random_state):
    """DataFrame with one categorical column for GN/GNHF tests."""
    np.random.seed(random_state)
    n = 80
    y = np.concatenate([
        np.random.normal(0, 1, 60),
        np.random.normal(6, 0.5, 20),
    ])
    return pd.DataFrame({
        "a": np.random.randn(n),
        "cat": np.random.choice(["X", "Y", "Z"], size=n),
        "y": y,
    })
