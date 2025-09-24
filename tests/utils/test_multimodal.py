# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from unittest.mock import patch

import torch
from flow_matching.path.mixture import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler

from flow_matching.utils.multimodal import _default_continuous_loss, Flow
from torch import nn


class DummyContinuousPath:
    """Simple placeholder for a continuous path."""

    pass


class DummyModel(nn.Module):
    """Model that returns logits for discrete and scaled inputs for continuous."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, xs, t, **kwargs):
        outputs = []
        for x in xs:
            if x.dtype == torch.long:
                batch = x.shape[0]
                # Return random logits for discrete modality
                outputs.append(torch.randn(batch, self.num_classes))
            else:
                # Return a simple transformation for continuous modality
                outputs.append(x * 2.0)
        return outputs


class DummyMultimodalSolver:
    """Mock solver that records arguments and returns predefined samples."""

    def __init__(self, model, modality_configs):
        self.model = model
        self.modality_configs = modality_configs
        self.called_with = {}

    def sample(self, **kwargs):
        self.called_with = kwargs
        # Return a list of tensors matching the number of modalities
        return [torch.tensor([1.0]), torch.tensor([2.0])]


class TestFlow(unittest.TestCase):
    def setUp(self):
        self.num_classes = 5
        self.discrete_path = MixtureDiscreteProbPath(
            scheduler=PolynomialConvexScheduler(n=2.0)
        )
        self.continuous_path = DummyContinuousPath()
        self.modalities = {
            "disc": {"path": self.discrete_path},
            "cont": {"path": self.continuous_path},
        }
        self.model = DummyModel(num_classes=self.num_classes)
        self.flow = Flow(model=self.model, modalities=self.modalities)

    def test_init_paths_and_losses(self):
        # Paths should be stored correctly
        self.assertIn("disc", self.flow.paths)
        self.assertIn("cont", self.flow.paths)
        self.assertIs(self.flow.paths["disc"], self.discrete_path)
        self.assertIs(self.flow.paths["cont"], self.continuous_path)

        # Loss functions: discrete should be MixturePathGeneralizedKL (callable)
        self.assertTrue(callable(self.flow.loss_fns["disc"]))
        # Continuous should use the default continuous loss
        self.assertIs(self.flow.loss_fns["cont"], _default_continuous_loss)

    def test_training_loss_computation(self):
        batch = 3
        # Discrete tensors (int64)
        x1_disc = torch.randint(0, self.num_classes, (batch,))
        x_t_disc = torch.randint(0, self.num_classes, (batch,))
        # Continuous tensors (float32)
        x1_cont = torch.randn(batch, 2)
        x_t_cont = torch.randn(batch, 2)
        dx_t_cont = torch.randn(batch, 2)
        # Assemble inputs matching modality order (disc, cont)
        x_1 = [x1_disc, x1_cont]
        x_t = [x_t_disc, x_t_cont]
        dx_t = [None, dx_t_cont]
        t = [torch.rand(batch), torch.rand(batch)]

        total_loss, loss_dict = self.flow.training_loss(x_1, x_t, dx_t, t)

        # Total loss should be a scalar tensor
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertEqual(total_loss.dim(), 0)

        # Loss dict should contain both modalities
        self.assertSetEqual(set(loss_dict.keys()), {"disc", "cont"})
        # Each entry should be a scalar tensor
        for loss in loss_dict.values():
            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.dim(), 0)

        # Total loss should equal sum of individual losses
        summed = sum(loss for loss in loss_dict.values())
        self.assertTrue(torch.allclose(total_loss, summed))

    def test_training_loss_mismatched_lengths(self):
        batch = 2
        x1_disc = torch.randint(0, self.num_classes, (batch,))
        x_t_disc = torch.randint(0, self.num_classes, (batch,))
        # x1_cont = torch.randn(batch, 2)
        # x_t_cont = torch.randn(batch, 2)
        # dx_t_cont = torch.randn(batch, 2)

        # Omit the continuous modality to trigger assertion
        x_1 = [x1_disc]
        x_t = [x_t_disc]
        dx_t = [None]
        t = [torch.rand(batch)]

        with self.assertRaises(AssertionError):
            self.flow.training_loss(x_1, x_t, dx_t, t)

    def test_sample_dtype_validation_and_output(self):
        batch = 4
        # Correct dtypes
        x_init_disc = torch.randint(0, self.num_classes, (batch,))
        x_init_cont = torch.randn(batch, 2)

        with patch(
            "flow_matching.utils.multimodal.MultimodalSolver",
            DummyMultimodalSolver,
        ):
            samples = self.flow.sample([x_init_disc, x_init_cont], steps=5)

        # Should receive the dummy solver's output
        self.assertEqual(len(samples), 2)
        self.assertTrue(torch.equal(samples[0], torch.tensor([1.0])))
        self.assertTrue(torch.equal(samples[1], torch.tensor([2.0])))

    def test_sample_wrong_dtype_raises(self):
        batch = 3
        # Wrong dtype for discrete modality (float instead of long)
        x_init_disc = torch.randn(batch, dtype=torch.float32)
        x_init_cont = torch.randn(batch, 2)

        with self.assertRaises(AssertionError):
            self.flow.sample([x_init_disc, x_init_cont], steps=5)

    def test_custom_loss_weights(self):
        # Define modalities with custom loss weights
        modalities = {
            "disc": {"path": self.discrete_path, "weight": 0.5},
            "cont": {"path": self.continuous_path, "weight": 2.0},
        }
        flow = Flow(model=self.model, modalities=modalities)

        # Prepare inputs
        batch = 3
        x1_disc = torch.randint(0, self.num_classes, (batch,))
        x_t_disc = torch.randint(0, self.num_classes, (batch,))
        x1_cont = torch.randn(batch, 2)
        x_t_cont = torch.randn(batch, 2)
        dx_t_cont = torch.randn(batch, 2)
        x_1 = [x1_disc, x1_cont]
        x_t = [x_t_disc, x_t_cont]
        dx_t = [None, dx_t_cont]
        t = [torch.rand(batch), torch.rand(batch)]

        total_loss, loss_dict = flow.training_loss(x_1, x_t, dx_t, t)

        # Compute expected weighted total loss
        expected_total = loss_dict["disc"] * 0.5 + loss_dict["cont"] * 2.0
        self.assertTrue(torch.allclose(total_loss, expected_total))

        # Verify that loss_weights are stored correctly
        self.assertEqual(flow.loss_weights["disc"], 0.5)
        self.assertEqual(flow.loss_weights["cont"], 2.0)


if __name__ == "__main__":
    unittest.main()
