import unittest

import numpy as np
import torch
import torch.nn as nn

from utils.infer_utils import _axis_starts, infer_patches


class ConstantRiceModel(nn.Module):
    n_classes = 5

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.seen_shapes = []

    def forward(self, inputs):
        self.seen_shapes.append(tuple(inputs.shape))
        batch, _, height, width = inputs.shape
        logits = torch.zeros(
            batch,
            self.n_classes,
            height,
            width,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        logits[:, 2] = 5.0
        return logits


class InferUtilsCheck(unittest.TestCase):
    def test_train_scale_sliding_windows_cover_real_inference_tile(self):
        self.assertEqual(_axis_starts(301, 140, 70), [0, 70, 140, 161])
        self.assertEqual(_axis_starts(320, 140, 70), [0, 70, 140, 180])

    def test_raw_patches_are_resized_and_batched(self):
        model = ConstantRiceModel()
        image = np.zeros((301, 320, 2), dtype=np.float32)

        output = infer_patches(
            model,
            torch.device("cpu"),
            image,
            patch_size=140,
            stride=70,
            model_input_size=32,
            patch_batch_size=3,
        )

        self.assertEqual(output.shape, (301, 320, 3))
        self.assertTrue(np.all(output == np.array([255, 0, 0], dtype=np.uint8)))
        self.assertEqual(sum(shape[0] for shape in model.seen_shapes), 16)
        self.assertTrue(all(shape[1:] == (2, 32, 32) for shape in model.seen_shapes))

    def test_patch_batch_size_must_be_positive(self):
        model = ConstantRiceModel()
        image = np.zeros((32, 32, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "patch_batch_size"):
            infer_patches(
                model,
                torch.device("cpu"),
                image,
                patch_size=16,
                model_input_size=32,
                patch_batch_size=0,
            )


if __name__ == "__main__":
    unittest.main()
