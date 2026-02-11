"""Unit tests for DataHandler input validation (via public resamplers)."""

import unittest

import numpy as np
import pandas as pd

import PyImbalReg as pir


class TestDataHandlerValidation(unittest.TestCase):
    """Validation of DataFrame, y_col, threshold, percentages, rel_func."""

    def setUp(self):
        self.valid_df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [0.1, 0.5, 0.9],
        })

    def test_rejects_non_dataframe(self):
        with self.assertRaises(TypeError):
            pir.RandomOversampling(
                df=[1, 2, 3],
                rel_func="default",
                threshold=0.7,
                o_percentage=2,
            )

    def test_rejects_dataframe_with_nan(self):
        df_nan = self.valid_df.copy()
        df_nan.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            pir.RandomOversampling(
                df=df_nan,
                rel_func="default",
                threshold=0.7,
                o_percentage=2,
            )

    def test_rejects_invalid_y_col_name_type(self):
        with self.assertRaises(TypeError):
            pir.RandomOversampling(
                df=self.valid_df,
                y_col_name=123,
                rel_func="default",
                threshold=0.7,
                o_percentage=2,
            )

    def test_rejects_y_col_name_not_in_columns(self):
        with self.assertRaises(ValueError):
            pir.RandomOversampling(
                df=self.valid_df,
                y_col_name="z",
                rel_func="default",
                threshold=0.7,
                o_percentage=2,
            )

    def test_rejects_threshold_not_float(self):
        with self.assertRaises(ValueError):
            pir.RandomOversampling(
                df=self.valid_df,
                rel_func="default",
                threshold="0.7",
                o_percentage=2,
            )

    def test_rejects_threshold_out_of_range(self):
        with self.assertRaises(ValueError):
            pir.RandomOversampling(
                df=self.valid_df,
                rel_func="default",
                threshold=1.0,
                o_percentage=2,
            )
        with self.assertRaises(ValueError):
            pir.RandomOversampling(
                df=self.valid_df,
                rel_func="default",
                threshold=0.0,
                o_percentage=2,
            )

    def test_rejects_o_percentage_not_above_one(self):
        with self.assertRaises(ValueError):
            pir.RandomOversampling(
                df=self.valid_df,
                rel_func="default",
                threshold=0.7,
                o_percentage=1,
            )

    def test_rejects_u_percentage_not_in_open_interval(self):
        with self.assertRaises(ValueError):
            pir.RandomUndersampling(
                df=self.valid_df,
                rel_func="default",
                threshold=0.7,
                u_percentage=0.0,
            )
        with self.assertRaises(ValueError):
            pir.RandomUndersampling(
                df=self.valid_df,
                rel_func="default",
                threshold=0.7,
                u_percentage=1.0,
            )

    def test_rejects_non_callable_rel_func(self):
        with self.assertRaises(TypeError):
            pir.RandomOversampling(
                df=self.valid_df,
                rel_func="not_a_function",
                threshold=0.7,
                o_percentage=2,
            )

    def test_rejects_rel_func_returning_outside_0_1(self):
        def bad_rel(x):
            return 1.5

        with self.assertRaises(ValueError):
            pir.RandomOversampling(
                df=self.valid_df,
                rel_func=bad_rel,
                threshold=0.7,
                o_percentage=2,
            )

    def test_accepts_default_rel_func_and_none_y_col_uses_last_column(self):
        ro = pir.RandomOversampling(
            df=self.valid_df,
            rel_func="default",
            threshold=0.7,
            o_percentage=2,
        )
        self.assertEqual(ro.y_col_name, "y")

    def test_accepts_explicit_y_col_name(self):
        ro = pir.RandomOversampling(
            df=self.valid_df,
            y_col_name="y",
            rel_func="default",
            threshold=0.7,
            o_percentage=2,
        )
        self.assertEqual(ro.y_col_name, "y")


class TestGNHFValidation(unittest.TestCase):
    """GNHF-specific validation: rel_func must be None."""

    def setUp(self):
        self.df = pd.DataFrame({
            "x": np.random.randn(50),
            "y": np.random.randn(50),
        })

    def test_gnhf_rejects_rel_func(self):
        with self.assertRaises(ValueError):
            pir.GNHF(
                df=self.df,
                rel_func="default",
                bins=5,
                perm_amp=0.1,
            )

    def test_gnhf_accepts_rel_func_none(self):
        gnhf = pir.GNHF(
            df=self.df,
            rel_func=None,
            bins=5,
            perm_amp=0.1,
            categorical_columns=[],
            random_state=42,
        )
        result = gnhf.get()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["x", "y"])
