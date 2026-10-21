import unittest

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class ModelShapeTests(unittest.TestCase):
    def test_csrnet_output_is_one_eighth(self):
        from src.models.csrnet import CSRNet
        out = CSRNet(pretrained=False)(torch.randn(1, 3, 64, 96))
        self.assertEqual(tuple(out.shape), (1, 1, 8, 12))

    def test_freeze_frontend(self):
        from src.models.csrnet import CSRNet
        m = CSRNet(pretrained=False, freeze_frontend=True)
        self.assertFalse(any(p.requires_grad for p in m.frontend.parameters()))
        self.assertTrue(all(p.requires_grad for p in m.backend.parameters()))

    def test_baselines_output_one_value_per_image(self):
        from src.models import build_model
        for name in ("mlp", "cnn"):
            for act in ("relu", "tanh", "sigmoid"):
                out = build_model(name, activation=act, dropout=0.3)(torch.randn(2, 3, 128, 128))
                self.assertEqual(tuple(out.shape), (2,))

    def test_unknown_model_rejected(self):
        from src.models import build_model
        with self.assertRaises(ValueError):
            build_model("resnet")


if __name__ == "__main__":
    unittest.main()
