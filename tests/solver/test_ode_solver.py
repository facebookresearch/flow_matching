# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from unittest.mock import MagicMock

import torch
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from torch import Tensor


class DummyModel(ModelWrapper):
    def __init__(self):
        super().__init__(None)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        return (x * 0.0 + 1.0) * 3.0 * t**2


class ConstantVelocityModel(ModelWrapper):
    def __init__(self):
        super().__init__(None)
        self.a = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        return x * 0.0 + self.a


class TestODESolver(unittest.TestCase):
    def setUp(self):
        self.mock_velocity_model = MagicMock(spec=ModelWrapper)
        self.mock_velocity_model.return_value = torch.tensor([1.0, 1.0])

        self.dummy_velocity_model = DummyModel()
        self.constant_velocity_model = ConstantVelocityModel()

        # Initialize the ODESolver with the mock model
        self.mock_solver = ODESolver(velocity_model=self.mock_velocity_model)
        self.dummy_solver = ODESolver(velocity_model=self.dummy_velocity_model)
        self.constant_velocity_solver = ODESolver(
            velocity_model=self.constant_velocity_model
        )

    def test_sample(self):
        x_init = torch.tensor([0.0, 0.0])
        step_size = 0.1
        time_grid = torch.tensor([0.0, 1.0])

        result = self.mock_solver.sample(
            x_init=x_init, step_size=step_size, time_grid=time_grid
        )

        self.assertIsInstance(result, Tensor)
        self.mock_velocity_model.assert_called()
        self.assertEqual(x_init.shape, result.shape)

    def test_sample_with_different_methods(self):
        x_init = torch.tensor([1.0, 0.0])
        step_size = 0.001
        time_grid = torch.tensor([0.0, 1.0])

        for method in ["euler", "dopri5", "midpoint", "heun3"]:
            with self.subTest(method=method):
                result = self.dummy_solver.sample(
                    x_init=x_init,
                    step_size=step_size if method != "dopri5" else None,
                    time_grid=time_grid,
                    method=method,
                )
                self.assertIsInstance(result, Tensor)
                self.assertTrue(
                    torch.allclose(torch.tensor([2.0, 1.0]), result, atol=1e-2),
                    "The solution to the velocity field 3t^3 from 0 to 1 is incorrect.",
                )

    def test_gradients(self):
        x_init = torch.tensor([1.0, 0.0])
        step_size = 0.001
        time_grid = torch.tensor([0.0, 1.0])

        for method in ["euler", "dopri5", "midpoint", "heun3"]:
            with self.subTest(method=method):
                self.constant_velocity_model.zero_grad()
                result = self.constant_velocity_solver.sample(
                    x_init=x_init,
                    step_size=step_size if method != "dopri5" else None,
                    time_grid=time_grid,
                    method=method,
                    enable_grad=True,
                )
                loss = result.sum()
                loss.backward()
                self.assertAlmostEqual(
                    self.constant_velocity_model.a.grad, 2.0, delta=1e-4
                )

    def test_no_gradients(self):
        x_init = torch.tensor([1.0, 0.0])
        step_size = 0.001
        time_grid = torch.tensor([0.0, 1.0])

        method = "euler"
        self.constant_velocity_model.zero_grad()
        result = self.constant_velocity_solver.sample(
            x_init=x_init,
            step_size=step_size,
            time_grid=time_grid,
            method=method,
        )
        loss = result.sum()

        with self.assertRaises(RuntimeError):
            loss.backward()

    def test_compute_likelihood(self):
        x_1 = torch.tensor([[0.0, 0.0]])
        step_size = 0.1

        # Define a dummy log probability function
        def dummy_log_p(x: Tensor) -> Tensor:
            return -0.5 * torch.sum(x**2, dim=1)

        _, log_likelihood = self.dummy_solver.compute_likelihood(
            x_1=x_1,
            log_p0=dummy_log_p,
            step_size=step_size,
            exact_divergence=False,
        )
        self.assertIsInstance(log_likelihood, Tensor)
        self.assertEqual(x_1.shape[0], log_likelihood.shape[0])

        with self.assertRaises(RuntimeError):
            log_likelihood.backward()

    def test_compute_likelihood_exact_divergence(self):
        x_1 = torch.tensor([[0.0, 0.0]], requires_grad=True)
        step_size = 0.1

        # Define a dummy log probability function
        def dummy_log_p(x: Tensor) -> Tensor:
            return -0.5 * torch.sum(x**2)

        x_0, log_likelihood = self.constant_velocity_solver.compute_likelihood(
            x_1=x_1,
            log_p0=dummy_log_p,
            step_size=step_size,
            exact_divergence=True,
            enable_grad=True,
        )
        self.assertIsInstance(log_likelihood, Tensor)
        self.assertEqual(x_1.shape[0], log_likelihood.shape[0])
        self.assertTrue(
            torch.allclose(dummy_log_p(x_1 - 1.0), log_likelihood, atol=1e-2),
        )
        self.assertTrue(
            torch.allclose(x_1 - 1.0, x_0, atol=1e-2),
        )
        log_likelihood.backward()
        self.assertTrue(
            torch.allclose(x_1.grad, torch.tensor([1.0, 1.0]), atol=1e-2),
        )


if __name__ == "__main__":
    unittest.main()
