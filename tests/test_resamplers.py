"""Unit tests for resampling methods: RO, RU, GN, WERCS, GNHF."""

import unittest

import numpy as np
import pandas as pd

import PyImbalReg as pir


class TestRandomOversampling(unittest.TestCase):
    """RandomOversampling.get() returns DataFrame with expected properties."""

    def test_get_returns_dataframe(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0.1, 0.5, 10.0]})
        ro = pir.RandomOversampling(
            df=df,
            rel_func="default",
            threshold=0.7,
            o_percentage=3,
            random_state=42,
        )
        result = ro.get()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["x", "y"])

    def test_oversampling_increases_size_when_rare_exists(self):
        df = pd.DataFrame({
            "x": list(range(20)),
            "y": [0.1] * 18 + [5.0, 6.0],
        })
        ro = pir.RandomOversampling(
            df=df,
            rel_func="default",
            threshold=0.7,
            o_percentage=4,
            random_state=42,
        )
        result = ro.get()
        self.assertGreaterEqual(len(result), len(df))

    def test_reproducibility_with_random_state(self):
        np.random.seed(1)
        n_common, n_rare = 25, 5
        df = pd.DataFrame({
            "x": np.random.randn(n_common + n_rare),
            "y": np.concatenate([
                np.random.randn(n_common),
                [10.0, 11.0, 12.0, 13.0, 14.0],
            ]),
        })
        ro1 = pir.RandomOversampling(
            df=df,
            rel_func="default",
            threshold=0.7,
            o_percentage=3,
            random_state=99,
        )
        ro2 = pir.RandomOversampling(
            df=df,
            rel_func="default",
            threshold=0.7,
            o_percentage=3,
            random_state=99,
        )
        out1 = ro1.get()
        out2 = ro2.get()
        pd.testing.assert_frame_equal(out1, out2)


class TestRandomUndersampling(unittest.TestCase):
    """RandomUndersampling.get() returns DataFrame with expected properties."""

    def test_get_returns_dataframe(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0.1, 0.5, 10.0]})
        ru = pir.RandomUndersampling(
            df=df,
            rel_func="default",
            threshold=0.7,
            u_percentage=0.5,
            random_state=42,
        )
        result = ru.get()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["x", "y"])

    def test_undersampling_reduces_or_keeps_size(self):
        df = pd.DataFrame({
            "x": list(range(50)),
            "y": [0.1] * 40 + [5.0] * 10,
        })
        ru = pir.RandomUndersampling(
            df=df,
            rel_func="default",
            threshold=0.7,
            u_percentage=0.5,
            random_state=42,
        )
        result = ru.get()
        self.assertLessEqual(len(result), len(df))

    def test_reproducibility_with_random_state(self):
        df = pd.DataFrame({
            "x": np.random.randn(40),
            "y": np.concatenate([np.random.randn(35), [8.0] * 5]),
        })
        ru1 = pir.RandomUndersampling(
            df=df,
            rel_func="default",
            threshold=0.7,
            u_percentage=0.5,
            random_state=7,
        )
        ru2 = pir.RandomUndersampling(
            df=df,
            rel_func="default",
            threshold=0.7,
            u_percentage=0.5,
            random_state=7,
        )
        pd.testing.assert_frame_equal(ru1.get(), ru2.get())


class TestGaussianNoise(unittest.TestCase):
    """GaussianNoise.get() returns DataFrame with expected properties."""

    def test_get_returns_dataframe(self):
        # Need enough points so at least one "rare" bin exists for GN oversampling
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 4 + [10.0, 11.0],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5] * 4 + [9.0, 10.0],
        })
        gn = pir.GaussianNoise(
            df=df,
            rel_func="default",
            threshold=0.7,
            o_percentage=2,
            u_percentage=0.5,
            perm_amp=0.1,
            categorical_columns=[],
            random_state=42,
        )
        result = gn.get()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["x", "y"])

    def test_reproducibility_with_random_state(self):
        df = pd.DataFrame({
            "x": np.random.randn(30),
            "y": np.concatenate([np.random.randn(25), [10.0, 11.0, 12.0, 13.0, 14.0]]),
        })
        gn1 = pir.GaussianNoise(
            df=df,
            rel_func="default",
            threshold=0.7,
            o_percentage=2.5,
            u_percentage=0.5,
            perm_amp=0.1,
            categorical_columns=[],
            random_state=123,
        )
        gn2 = pir.GaussianNoise(
            df=df,
            rel_func="default",
            threshold=0.7,
            o_percentage=2.5,
            u_percentage=0.5,
            perm_amp=0.1,
            categorical_columns=[],
            random_state=123,
        )
        pd.testing.assert_frame_equal(gn1.get(), gn2.get())


class TestWERCS(unittest.TestCase):
    """WERCS.get() returns combined DataFrame."""

    def test_get_returns_dataframe(self):
        df = pd.DataFrame({
            "x": np.random.randn(20),
            "y": np.random.randn(20),
        })
        wercs = pir.WERCS(
            df=df,
            rel_func="default",
            threshold=0.7,
            o_percentage=2,
            u_percentage=0.5,
            random_state=42,
        )
        result = wercs.get()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["x", "y"])
        self.assertGreater(len(result), len(df))

    def test_reproducibility_with_random_state(self):
        df = pd.DataFrame({
            "x": np.random.randn(25),
            "y": np.random.randn(25),
        })
        w1 = pir.WERCS(
            df=df,
            rel_func="default",
            threshold=0.6,
            o_percentage=2,
            u_percentage=0.6,
            random_state=0,
        )
        w2 = pir.WERCS(
            df=df,
            rel_func="default",
            threshold=0.6,
            o_percentage=2,
            u_percentage=0.6,
            random_state=0,
        )
        pd.testing.assert_frame_equal(w1.get(), w2.get())


class TestGNHF(unittest.TestCase):
    """GNHF.get() with histogram-based balancing."""

    def test_get_returns_dataframe(self):
        np.random.seed(42)
        # Use 3 bins and enough points per bin so no bin has 0 or 1 sample
        y = np.concatenate([
            np.random.uniform(0, 1, 40),
            np.random.uniform(2, 3, 40),
            np.random.uniform(5, 6, 40),
        ])
        df = pd.DataFrame({
            "x": np.random.randn(120),
            "y": y,
        })
        gnhf = pir.GNHF(
            df=df,
            rel_func=None,
            bins=3,
            perm_amp=0.1,
            categorical_columns=[],
            random_state=42,
        )
        result = gnhf.get()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["x", "y"])

    def test_raises_when_bin_has_zero_or_one_sample(self):
        # Very few points and many bins -> some bins empty or single
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
        })
        gnhf = pir.GNHF(
            df=df,
            rel_func=None,
            bins=10,
            perm_amp=0.1,
            categorical_columns=[],
            random_state=42,
        )
        with self.assertRaises(ValueError):
            gnhf.get()
