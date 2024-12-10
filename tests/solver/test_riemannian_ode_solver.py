# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from flow_matching.solver.riemannian_ode_solver import RiemannianODESolver
from flow_matching.utils.manifolds import Sphere


class HundredVelocityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        return torch.ones_like(x) * 100.0


class ZeroVelocityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        return torch.zeros_like(x)


class TestRiemannianODESolver(unittest.TestCase):
    def setUp(self):
        self.manifold = Sphere()
        self.velocity_model = HundredVelocityModel()
        self.solver = RiemannianODESolver(self.manifold, self.velocity_model)

    def test_init(self):
        self.assertEqual(self.solver.manifold, self.manifold)
        self.assertEqual(self.solver.velocity_model, self.velocity_model)

    def test_sample_euler(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        result = self.solver.sample(
            x_init, step_size, method="euler", time_grid=time_grid
        )
        self.assertTrue(
            torch.allclose(
                result,
                torch.nn.functional.normalize(
                    torch.tensor([1.0, 1.0, 1.0]), dim=0, p=2.0
                ),
                rtol=1e-3,
            )
        )

    def test_sample_midpoint(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        result = self.solver.sample(
            x_init, step_size, method="midpoint", time_grid=time_grid
        )
        self.assertTrue(
            torch.allclose(
                result,
                torch.nn.functional.normalize(
                    torch.tensor([1.0, 1.0, 1.0]), dim=0, p=2.0
                ),
                rtol=1e-3,
            )
        )

    def test_sample_rk4(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        result = self.solver.sample(
            x_init, step_size, method="rk4", time_grid=time_grid
        )
        self.assertTrue(
            torch.allclose(
                result,
                torch.nn.functional.normalize(
                    torch.tensor([1.0, 1.0, 1.0]), dim=0, p=2.0
                ),
                rtol=1e-3,
            )
        )

    def test_zero_velocity_euler(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        zero_velocity_model = ZeroVelocityModel()
        solver = RiemannianODESolver(self.manifold, zero_velocity_model)
        result = solver.sample(x_init, step_size, method="euler", time_grid=time_grid)
        self.assertTrue(torch.allclose(result, x_init))

    def test_zero_velocity_midpoint(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        zero_velocity_model = ZeroVelocityModel()
        solver = RiemannianODESolver(self.manifold, zero_velocity_model)
        result = solver.sample(
            x_init, step_size, method="midpoint", time_grid=time_grid
        )
        self.assertTrue(torch.allclose(result, x_init))

    def test_zero_velocity_rk4(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        zero_velocity_model = ZeroVelocityModel()
        solver = RiemannianODESolver(self.manifold, zero_velocity_model)
        result = solver.sample(x_init, step_size, method="rk4", time_grid=time_grid)
        self.assertTrue(torch.allclose(result, x_init))

    def test_sample_euler_step_size_none(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        time_grid = torch.linspace(0.0, 1.0, steps=100)
        result = self.solver.sample(x_init, None, method="euler", time_grid=time_grid)
        self.assertTrue(
            torch.allclose(
                result,
                torch.nn.functional.normalize(
                    torch.tensor([1.0, 1.0, 1.0]), dim=0, p=2.0
                ),
                rtol=1e-3,
            )
        )

    def test_sample_euler_verbose(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 1.0])
        result = self.solver.sample(
            x_init, step_size, method="euler", time_grid=time_grid, verbose=True
        )
        self.assertTrue(isinstance(result, torch.Tensor))

    def test_sample_return_intermediates_euler(self):
        x_init = self.manifold.projx(torch.randn(1, 3))
        step_size = 0.01
        time_grid = torch.tensor([0.0, 0.5, 1.0])
        result = self.solver.sample(
            x_init,
            step_size,
            method="euler",
            time_grid=time_grid,
            return_intermediates=True,
        )
        self.assertEqual(result.shape, (3, 1, 3))  # Two intermediate points


if __name__ == "__main__":
    unittest.main()
