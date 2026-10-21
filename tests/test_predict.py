import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from src import predict as P


class PredictStubTests(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("SS_MODEL_PATH")
        os.environ["SS_MODEL_PATH"] = os.path.join(tempfile.mkdtemp(), "missing.onnx")
        self.img = (np.random.default_rng(1).random((120, 200, 3)) * 255).astype(np.uint8)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("SS_MODEL_PATH", None)
        else:
            os.environ["SS_MODEL_PATH"] = self._old

    def test_backend_is_stub_without_model(self):
        self.assertEqual(P.backend_name(), "stub")

    def test_contract_count_and_density(self):
        count, dens = P.predict(self.img)
        self.assertIsInstance(count, float)
        self.assertEqual(dens.shape, (15, 25))
        self.assertAlmostEqual(count, float(dens.sum()), places=2)
        self.assertGreater(count, 0)

    def test_deterministic_for_same_image(self):
        a = P.predict(self.img)
        b = P.predict(self.img.copy())
        self.assertEqual(a[0], b[0])
        self.assertTrue(np.array_equal(a[1], b[1]))

    def test_accepts_pil_and_path(self):
        count_np, _ = P.predict(self.img)
        count_pil, _ = P.predict(Image.fromarray(self.img))
        self.assertEqual(count_np, count_pil)
        path = os.path.join(tempfile.mkdtemp(), "x.png")
        Image.fromarray(self.img).save(path)
        count_path, _ = P.predict(path)
        self.assertAlmostEqual(count_np, count_path, places=2)

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            P.predict(np.zeros((5, 5, 2), dtype=np.uint8))
        with self.assertRaises(ValueError):
            P.predict("/no/such/file.jpg")

    def test_preprocess_shape_and_limits(self):
        big = np.zeros((1500, 2000, 3), dtype=np.uint8)
        x = P.preprocess(big)
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x.shape[0:2], (1, 3))
        self.assertEqual(x.shape[2] % 8, 0)
        self.assertEqual(x.shape[3] % 8, 0)
        self.assertLessEqual(max(x.shape[2:]), 1024)

    def test_preprocess_normalises_like_imagenet(self):
        x = P.preprocess(np.full((16, 16, 3), 255, dtype=np.uint8))
        expected = (1.0 - P.MEAN) / P.STD
        self.assertTrue(np.allclose(x[0, :, 0, 0], expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
