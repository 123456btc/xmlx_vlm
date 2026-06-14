"""Smoke tests for DiffusionGemma model support."""

import unittest

from xmlx_vlm.models.diffusion_gemma import Model, ModelConfig, TextConfig
from xmlx_vlm.utils import get_model_and_args


class TestDiffusionGemma(unittest.TestCase):
    def test_model_type_registered(self):
        config = {"model_type": "diffusion_gemma"}
        arch, model_type = get_model_and_args(config)
        self.assertEqual(model_type, "diffusion_gemma")
        self.assertTrue(hasattr(arch, "Model"))
        self.assertTrue(hasattr(arch, "ModelConfig"))

    def test_config_has_canvas_length(self):
        config = ModelConfig(text_config=TextConfig())
        self.assertEqual(config.model_type, "diffusion_gemma")
        self.assertEqual(config.canvas_length, 256)
        self.assertIsNotNone(config.text_config)

    def test_model_can_be_instantiated(self):
        config = ModelConfig(text_config=TextConfig())
        model = Model(config)
        self.assertEqual(model.model_type, "diffusion_gemma")
        self.assertTrue(hasattr(model, "language_model"))


if __name__ == "__main__":
    unittest.main()
