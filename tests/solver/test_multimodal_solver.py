# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from unittest.mock import MagicMock

import torch
from flow_matching.path import AffineProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import CondOTScheduler, PolynomialConvexScheduler

from flow_matching.solver.multimodal_solver import MultimodalSolver
from flow_matching.utils import ModelWrapper
from torch import Tensor


# ----------------------------------------------------------------------
# Helper models for continuous and discrete modalities
# ----------------------------------------------------------------------
class ContinuousVelocityModel(ModelWrapper):
    def __init__(self):
        super().__init__(None)

    def forward(self, xs: list[Tensor], t: list[Tensor], **extras) -> list[Tensor]:
        # xs is a list of modality states; we only have one continuous modality here.
        # Return a list with the same length as xs.
        return [2.0 * xs[0]]


class DiscreteLogitsModel(ModelWrapper):
    def __init__(self, vocab_size: int):
        super().__init__(None)
        self.vocab_size = vocab_size

    def forward(self, xs: list[Tensor], t: list[Tensor], **extras) -> list[Tensor]:
        """Produce logits that give probability 1.0 to the last class."""
        batch = xs[0].shape[0]
        logits = torch.full((batch, self.vocab_size), -1e9, device=xs[0].device)
        logits[:, -1] = 1e9
        return [logits]


# ----------------------------------------------------------------------
# Test suite
# ----------------------------------------------------------------------
class TestMultimodalSolver(unittest.TestCase):
    def setUp(self):
        # Continuous modality config (no extra args needed)
        self.continuous_cfg = {
            "type": "continuous",
            "path": AffineProbPath(scheduler=CondOTScheduler()),
            "x_1_prediction": False,
        }

        # Discrete modality config
        self.vocab_size = 3
        self.discrete_path = MixtureDiscreteProbPath(
            scheduler=PolynomialConvexScheduler(n=2.0)
        )
        self.discrete_cfg = {
            "type": "discrete",
            "path": self.discrete_path,
        }

        # Source distribution for divergence‑free term (uniform)
        self.source_p = torch.tensor([1.0 / self.vocab_size] * self.vocab_size)

        # Dummy models
        self.continuous_model = ContinuousVelocityModel()
        self.discrete_model = DiscreteLogitsModel(vocab_size=self.vocab_size)

        # Combined model that forwards to the appropriate sub‑model
        class CombinedModel(ModelWrapper):
            def __init__(self, cont, disc):
                super().__init__(None)
                self.cont = cont
                self.disc = disc

            def forward(self, xs, t, **extras):
                # xs[0] -> continuous, xs[1] -> discrete
                cont_out = self.cont.forward([xs[0]], t, **extras)[0]
                disc_out = self.disc.forward([xs[1]], t, **extras)[0]
                return [cont_out, disc_out]

        self.model = CombinedModel(self.continuous_model, self.discrete_model)

    # ------------------------------------------------------------------
    # Basic initialization test
    # ------------------------------------------------------------------
    def test_init(self):
        solver = MultimodalSolver(
            model=self.model,
            modality_configs=[self.continuous_cfg, self.discrete_cfg],
            source_distribution_p=self.source_p,
        )
        self.assertIs(solver.model, self.model)
        self.assertEqual(
            solver.modality_configs, [self.continuous_cfg, self.discrete_cfg]
        )
        self.assertTrue(torch.allclose(solver.source_distribution_p, self.source_p))

    # ------------------------------------------------------------------
    # Simple sampling test (continuous + discrete)
    # ------------------------------------------------------------------
    def test_sample_basic(self):
        solver = MultimodalSolver(
            model=self.model,
            modality_configs=[self.continuous_cfg, self.discrete_cfg],
            source_distribution_p=self.source_p,
        )
        # Initial states: continuous (batch=1, dim=1), discrete (batch=1, categorical)
        x_cont = torch.tensor([[0.0]])  # shape (1, 1)
        x_disc = torch.tensor([[0]])  # shape (1, 1)
        result = solver.sample(
            x_init=[x_cont, x_disc],
            step_size=0.1,
            time_grid=torch.tensor([0.0, 1.0]),
        )
        # Continuous modality: v = 2*x, Euler step => x_final = x0 + h*2*x0 = 0
        # Discrete modality: logits always select last class => final state = vocab_size-1
        self.assertTrue(torch.allclose(result[0], torch.zeros_like(result[0])))
        self.assertTrue(torch.equal(result[1], torch.tensor([self.vocab_size - 1])))

    # ------------------------------------------------------------------
    # Return intermediates test
    # ------------------------------------------------------------------
    def test_return_intermediates(self):
        solver = MultimodalSolver(
            model=self.model,
            modality_configs=[self.continuous_cfg, self.discrete_cfg],
            source_distribution_p=self.source_p,
        )
        x_cont = torch.tensor([[1.0]])  # start at 1.0
        x_disc = torch.tensor([[0]])  # start at class 0
        intermediates = solver.sample(
            x_init=[x_cont, x_disc],
            step_size=0.5,
            time_grid=torch.tensor([0.0, 0.5, 1.0]),
            return_intermediates=True,
        )
        # Should return a list of two lists (one per modality)
        self.assertEqual(len(intermediates), 2)
        # Continuous trajectory should have three entries (including start & end)
        self.assertEqual(len(intermediates[0]), 3)
        # Discrete trajectory should also have three entries
        self.assertEqual(len(intermediates[1]), 3)
        # Verify the final discrete state is the last class
        self.assertTrue(
            torch.equal(intermediates[1][-1], torch.tensor([self.vocab_size - 1]))
        )

    # ------------------------------------------------------------------
    # Gradient tracking test
    # ------------------------------------------------------------------
    def test_gradient_enabled(self):
        solver = MultimodalSolver(
            model=self.model,
            modality_configs=[self.continuous_cfg, self.discrete_cfg],
            source_distribution_p=self.source_p,
        )
        x_cont = torch.tensor([[2.0]], requires_grad=True)
        x_disc = torch.tensor([[0]], requires_grad=False)
        result = solver.sample(
            x_init=[x_cont, x_disc],
            step_size=0.1,
            time_grid=torch.tensor([0.0, 1.0]),
            enable_grad=True,
        )
        # Only the continuous modality should have a gradient
        loss = result[0].sum()
        loss.backward()
        self.assertIsNotNone(x_cont.grad)
        self.assertIsNone(x_disc.grad)

    # ------------------------------------------------------------------
    # Divergence‑free term test (non‑zero)
    # ------------------------------------------------------------------
    def test_divergence_free(self):
        # Use a mock model that returns zero logits for the discrete modality
        mock_model = MagicMock()
        mock_model.return_value = [
            torch.zeros(1, 1),
            torch.zeros(1, 1, self.vocab_size),
        ]

        solver = MultimodalSolver(
            model=mock_model,
            modality_configs=[self.continuous_cfg, self.discrete_cfg],
            source_distribution_p=self.source_p,
        )
        x_cont = torch.tensor([[0.0]])
        x_disc = torch.tensor([[0]])
        # Use a constant divergence‑free term
        result = solver.sample(
            x_init=[x_cont, x_disc],
            step_size=0.1,
            div_free=0.5,
            time_grid=torch.tensor([0.0, 1.0]),
        )
        # With a non‑zero div_free, the solver should not raise an assertion.
        # The exact numeric value is not critical; we just ensure the call succeeds.
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    # ------------------------------------------------------------------
    # Error handling tests
    # ------------------------------------------------------------------
    def test_mismatched_initial_states(self):
        solver = MultimodalSolver(
            model=self.model,
            modality_configs=[self.continuous_cfg, self.discrete_cfg],
        )
        # Provide only one initial state instead of two
        with self.assertRaises(ValueError):
            solver.sample(
                x_init=[torch.tensor([[0.0]])],
                step_size=0.1,
                time_grid=torch.tensor([0.0, 1.0]),
            )

    def test_invalid_modality_type(self):
        # Create a bad config list
        bad_cfg = [{"type": "unknown"}]
        with self.assertRaises(ValueError):
            MultimodalSolver(
                model=self.model,
                modality_configs=bad_cfg,
            )

    def test_missing_path_for_discrete(self):
        bad_cfg = [{"type": "discrete"}]  # No 'path' key
        with self.assertRaises(ValueError):
            MultimodalSolver(
                model=self.model,
                modality_configs=bad_cfg,
            )

    def test_non_callable_model(self):
        with self.assertRaises(TypeError):
            MultimodalSolver(
                model=123,  # Not callable
                modality_configs=[self.continuous_cfg],
            )


if __name__ == "__main__":
    unittest.main()
