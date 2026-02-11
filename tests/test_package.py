"""Package-level tests: imports, __all__, version."""

import unittest

import PyImbalReg as pir


class TestPackage(unittest.TestCase):
    """Public API and metadata."""

    def test_version_is_string(self):
        self.assertIsInstance(pir.__version__, str)
        self.assertRegex(pir.__version__, r"^\d+\.\d+\.\d+$")

    def test_all_exports(self):
        expected = {
            "DataHandler",
            "GNHF",
            "GaussianNoise",
            "RandomOversampling",
            "RandomUndersampling",
            "WERCS",
            "train_test_split",
            "__version__",
        }
        self.assertEqual(set(pir.__all__), expected)

    def test_can_import_public_classes_and_function(self):
        self.assertIsNotNone(pir.DataHandler)
        self.assertIsNotNone(pir.RandomOversampling)
        self.assertIsNotNone(pir.RandomUndersampling)
        self.assertIsNotNone(pir.GaussianNoise)
        self.assertIsNotNone(pir.WERCS)
        self.assertIsNotNone(pir.GNHF)
        self.assertIsNotNone(pir.train_test_split)
        self.assertTrue(callable(pir.train_test_split))

    def test_resamplers_have_get_method(self):
        self.assertTrue(hasattr(pir.RandomOversampling, "get"))
        self.assertTrue(hasattr(pir.RandomUndersampling, "get"))
        self.assertTrue(hasattr(pir.GaussianNoise, "get"))
        self.assertTrue(hasattr(pir.WERCS, "get"))
        self.assertTrue(hasattr(pir.GNHF, "get"))
