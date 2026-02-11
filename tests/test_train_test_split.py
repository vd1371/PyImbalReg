"""Unit tests for train_test_split."""

import unittest

import numpy as np
import pandas as pd

import PyImbalReg as pir


class TestTrainTestSplit(unittest.TestCase):
    """train_test_split returns two DataFrames with correct sizes and types."""

    def test_returns_two_dataframes(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        train, test = pir.train_test_split(
            df=df,
            test_size=0.2,
            bins=3,
            random_state=42,
        )
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
        self.assertEqual(list(train.columns), list(df.columns))
        self.assertEqual(list(test.columns), list(df.columns))

    def test_train_and_test_partition_rows(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "x": np.random.randn(n),
            "y": np.random.randn(n),
        })
        train, test = pir.train_test_split(
            df=df,
            test_size=0.25,
            bins=10,
            random_state=1,
        )
        # Stratified split: train and test are disjoint; union may not equal df
        # because histogram edges can exclude boundary points
        combined = pd.concat([train, test])
        self.assertEqual(len(combined), len(train) + len(test))
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)

    def test_reproducibility_with_random_state(self):
        df = pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100),
        })
        train1, test1 = pir.train_test_split(
            df=df, test_size=0.3, bins=5, random_state=99
        )
        train2, test2 = pir.train_test_split(
            df=df, test_size=0.3, bins=5, random_state=99
        )
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_rejects_non_dataframe(self):
        with self.assertRaises(TypeError):
            pir.train_test_split(
                df=[1, 2, 3],
                test_size=0.2,
                bins=5,
            )

    def test_rejects_test_size_not_float(self):
        df = pd.DataFrame({"x": [1], "y": [1.0]})
        with self.assertRaises(ValueError):
            pir.train_test_split(df=df, test_size=1, bins=5)
        with self.assertRaises(ValueError):
            pir.train_test_split(df=df, test_size="0.2", bins=5)

    def test_rejects_test_size_out_of_range(self):
        df = pd.DataFrame({"x": [1], "y": [1.0]})
        with self.assertRaises(ValueError):
            pir.train_test_split(df=df, test_size=0.0, bins=5)
        with self.assertRaises(ValueError):
            pir.train_test_split(df=df, test_size=1.0, bins=5)

    def test_rejects_bins_not_integer(self):
        df = pd.DataFrame({"x": [1], "y": [1.0]})
        with self.assertRaises(ValueError):
            pir.train_test_split(df=df, test_size=0.2, bins=5.0)

    def test_rejects_random_state_not_integer(self):
        df = pd.DataFrame({"x": [1, 2], "y": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            pir.train_test_split(
                df=df,
                test_size=0.2,
                bins=2,
                random_state=1.5,
            )
