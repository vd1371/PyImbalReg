"""Stratified train/test split for regression.

Splits a DataFrame so train and test have similar target distributions,
addressing imbalance that sklearn.model_selection.train_test_split ignores.
"""

import numpy as np
import pandas as pd


def train_test_split(
    df=pd.DataFrame(),
    test_size=None,
    bins=None,
    random_state=None,
):
    """Split a DataFrame into train and test with similar target distributions.

    Args:
        df: DataFrame with the last column as the target.
        test_size: Fraction of data for the test set (0 < test_size < 1).
        bins: Number of bins for stratifying by target (unused but kept for API).
        random_state: Seed for reproducible sampling.

    Returns:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "PyImbalReg train_test_split only supports pandas DataFrames."
        )
    if not isinstance(test_size, float):
        raise ValueError("test_size must be a float.")
    if not (0 < test_size < 1):
        raise ValueError("test_size must be in (0, 1).")
    if random_state is not None and not isinstance(random_state, int):
        raise ValueError("random_state must be an integer or None.")
    if not isinstance(bins, int):
        raise ValueError("bins must be an integer.")

    y = df.iloc[:, -1].values
    _, bin_edges = np.histogram(y)

    test_dfs_list = []
    train_dfs_list = []

    for left_edge, right_edge in zip(bin_edges[:-1], bin_edges[1:]):
        bin_df = df[(y > left_edge) & (y < right_edge)]
        test_df = bin_df.sample(frac=test_size, random_state=random_state)
        train_df = bin_df.drop(test_df.index)
        test_dfs_list.append(test_df)
        train_dfs_list.append(train_df)

    test_df = pd.concat(test_dfs_list)
    train_df = pd.concat(train_dfs_list)
    return train_df, test_df
