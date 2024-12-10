# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver


class DummyModel(torch.nn.Module):
    def forward(self, x, t, **extras):
        return torch.stack(
            [torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)], dim=-1
        )


class TestMixtureDiscreteEulerSolver(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))
        self.vocabulary_size = 3
        self.source_distribution_p = torch.tensor([0.5, 0.5, 0.0])

    def test_init(self):
        solver = MixtureDiscreteEulerSolver(
            model=self.model,
            path=self.path,
            vocabulary_size=self.vocabulary_size,
            source_distribution_p=self.source_distribution_p,
        )
        self.assertEqual(solver.model, self.model)
        self.assertEqual(solver.path, self.path)
        self.assertEqual(solver.vocabulary_size, self.vocabulary_size)
        self.assertTrue(
            torch.allclose(solver.source_distribution_p, self.source_distribution_p)
        )

    def test_sample(self):
        solver = MixtureDiscreteEulerSolver(
            model=self.model,
            path=self.path,
            vocabulary_size=self.vocabulary_size,
            source_distribution_p=self.source_distribution_p,
        )
        x_init = torch.tensor([[0]])
        step_size = 0.1
        time_grid = torch.tensor([0.0, 1.0])
        result = solver.sample(x_init, step_size, time_grid=time_grid)
        self.assertEqual(result, torch.ones_like(result) * 2)

    def test_sample_with_sym_term(self):
        solver = MixtureDiscreteEulerSolver(
            model=self.model,
            path=self.path,
            vocabulary_size=self.vocabulary_size,
            source_distribution_p=self.source_distribution_p,
        )
        x_init = torch.tensor([[0]])
        step_size = 0.1
        time_grid = torch.tensor([0.0, 1.0])
        div_free = 1.0
        result = solver.sample(
            x_init, step_size, time_grid=time_grid, div_free=div_free, verbose=True
        )
        self.assertIsInstance(result, torch.Tensor)
        result = solver.sample(
            x_init, step_size, time_grid=time_grid, div_free=lambda t: 1.0, verbose=True
        )
        self.assertIsInstance(result, torch.Tensor)

    def test_init_p_none(self):
        solver = MixtureDiscreteEulerSolver(
            model=self.model,
            path=self.path,
            vocabulary_size=self.vocabulary_size,
        )
        self.assertIsNone(solver.source_distribution_p)

    def test_sample_time_grid(self):
        solver = MixtureDiscreteEulerSolver(
            model=self.model,
            path=self.path,
            vocabulary_size=self.vocabulary_size,
            source_distribution_p=self.source_distribution_p,
        )
        x_init = torch.tensor([0])
        time_grid = torch.linspace(0.0, 1.0, steps=11)
        result = solver.sample(
            x_init, step_size=None, time_grid=time_grid, return_intermediates=True
        )
        self.assertEqual(result[-1], torch.ones_like(result[-1]) * 2)
        self.assertEqual(result.shape, (11, 1))

    def test_sample_return_intermediate(self):
        solver = MixtureDiscreteEulerSolver(
            model=self.model,
            path=self.path,
            vocabulary_size=self.vocabulary_size,
            source_distribution_p=self.source_distribution_p,
        )
        x_init = torch.tensor([0])
        time_grid = torch.linspace(0.0, 1.0, steps=3)
        result = solver.sample(
            x_init, step_size=0.1, time_grid=time_grid, return_intermediates=True
        )
        self.assertEqual(result[-1], torch.ones_like(result[-1]) * 2)
        self.assertEqual(result.shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
