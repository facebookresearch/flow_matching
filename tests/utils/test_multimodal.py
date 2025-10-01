# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from unittest.mock import patch

import torch
from flow_matching.path import AffineProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import CondOTScheduler, PolynomialConvexScheduler

from flow_matching.utils.multimodal import _default_continuous_loss, Flow
from torch import nn


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

    def __init__(self, model, modality_configs, model_sampling_fn=None):
        self.model = model
        self.modality_configs = modality_configs
        self.model_sampling_fn = model_sampling_fn
        self.called_with = {}

    def sample(self, **kwargs):
        self.called_with = kwargs
        # Return a list of tensors matching the number of modalities
        return [torch.tensor([1]), torch.tensor([2.0])]


class TestFlow(unittest.TestCase):
    def setUp(self):
        self.num_classes = 5
        self.discrete_path = MixtureDiscreteProbPath(
            scheduler=PolynomialConvexScheduler(n=2.0)
        )
        self.continuous_path = AffineProbPath(scheduler=CondOTScheduler())
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
            self.assertEqual(loss.mean().dim(), 0)

        # Total loss should equal sum of individual losses
        summed = sum(loss.mean() for loss in loss_dict.values())
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
            self.flow = Flow(
                model=self.model, modalities=self.modalities
            )  # Reinitialize to use dummy solver
            samples = self.flow.sample([x_init_disc, x_init_cont], steps=5)

        # Should receive the dummy solver's output
        self.assertEqual(len(samples), 2)
        self.assertTrue(torch.equal(samples[0], torch.tensor([1])))
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
        expected_total = loss_dict["disc"].mean() + loss_dict["cont"].mean()
        self.assertTrue(torch.allclose(total_loss, expected_total))

        # Verify that loss_weights are stored correctly
        self.assertEqual(flow.loss_weights["disc"], 0.5)
        self.assertEqual(flow.loss_weights["cont"], 2.0)

    def test_training_loss_x1_prediction_true(self):
        # Define a custom continuous loss that returns the target tensor.
        def custom_continuous_loss(pred, target, reduction="none"):
            # Return the target directly to verify it's used.
            return target

        # Set up modalities with x_1_prediction enabled for the continuous path.
        modalities = {
            "disc": {"path": self.discrete_path},
            "cont": {
                "path": self.continuous_path,
                "loss": custom_continuous_loss,
                "x_1_prediction": True,
            },
        }
        flow = Flow(model=self.model, modalities=modalities)

        # Prepare inputs.
        batch = 3
        x1_disc = torch.randint(0, self.num_classes, (batch,))
        x_t_disc = torch.randint(0, self.num_classes, (batch,))
        x1_cont = torch.randn(batch, 2)
        x_t_cont = torch.randn(batch, 2)
        dx_t_cont = torch.randn(
            batch, 2
        )  # Should be ignored due to x_1_prediction=True
        x_1 = [x1_disc, x1_cont]
        x_t = [x_t_disc, x_t_cont]
        dx_t = [None, dx_t_cont]
        t = [torch.rand(batch), torch.rand(batch)]

        total_loss, loss_dict = flow.training_loss(x_1, x_t, dx_t, t)

        # The continuous loss should have used x1_cont as the target.
        self.assertTrue(torch.allclose(loss_dict["cont"], x1_cont))
        # Total loss should be sum of discrete loss mean and x1_cont mean.
        expected_total = loss_dict["disc"].mean() + loss_dict["cont"].mean()
        self.assertTrue(torch.allclose(total_loss, expected_total))

    def test_training_loss_with_logits_argument(self):
        batch = 3
        # Discrete tensors (int64)
        x1_disc = torch.randint(0, self.num_classes, (batch,))
        x_t_disc = torch.randint(0, self.num_classes, (batch,))
        # Continuous tensors (float32)
        x1_cont = torch.randn(batch, 2)
        x_t_cont = torch.randn(batch, 2)
        dx_t_cont = torch.randn(batch, 2)
        x_1 = [x1_disc, x1_cont]
        x_t = [x_t_disc, x_t_cont]
        dx_t = [None, dx_t_cont]
        t = [torch.rand(batch), torch.rand(batch)]

        # Deterministic logits for discrete and continuous modalities
        logits_disc = torch.full((batch, self.num_classes), 0.5)
        logits_cont = torch.full_like(dx_t_cont, 0.1)
        logits = [logits_disc, logits_cont]

        # Ensure model forward is not called when logits are provided
        with patch.object(
            self.flow.model,
            "forward",
            side_effect=AssertionError("Model forward should not be called"),
        ):
            total_loss, loss_dict = self.flow.training_loss(
                x_1, x_t, dx_t, t, model_output=logits
            )

        # Verify total loss is scalar and matches sum of individual losses
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertEqual(total_loss.dim(), 0)
        self.assertSetEqual(set(loss_dict.keys()), {"disc", "cont"})
        summed = sum(loss.mean() for loss in loss_dict.values())
        self.assertTrue(torch.allclose(total_loss, summed))


if __name__ == "__main__":
    unittest.main()
