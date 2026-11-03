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

    def test_entrance_law(self):
        dense_p = torch.tensor([0.2, 0.3, 0.5])
        mask_p = torch.tensor([1.0, 0.0, 0.0])
        target = torch.tensor([0.0, 0.0, 1.0])
        N = 20000
        time_grid = torch.tensor([0.0, 0.3, 1.0])
        configs = [
            ("dense_linear_scalar", dense_p, 1.0, 1.0),
            ("mask_poly_scalar", mask_p, 2.0, 1.0),
            ("dense_cubic_smooth", dense_p, 3.0, None),
        ]
        for name, p, n, c in configs:
            with self.subTest(name=name):
                div_free = (lambda t: 1.0 + t) if c is None else c
                solver = MixtureDiscreteEulerSolver(
                    model=self.model,
                    path=MixtureDiscreteProbPath(
                        scheduler=PolynomialConvexScheduler(n=n)
                    ),
                    vocabulary_size=self.vocabulary_size,
                    source_distribution_p=p,
                )
                with torch.random.fork_rng():
                    torch.manual_seed(2024)
                    x_init = (
                        torch.multinomial(p, N * 2, replacement=True)
                        .reshape(N, 2)
                        .long()
                    )
                    result = solver.sample(
                        x_init,
                        step_size=None,
                        time_grid=time_grid,
                        return_intermediates=True,
                        div_free=div_free,
                        dtype_categorical=torch.float32,
                    )
                first = result[1]
                k_next = 0.3**n
                expected = (1.0 - k_next) * p + k_next * target
                flat = first.reshape(-1)
                counts = torch.bincount(flat, minlength=self.vocabulary_size).float()
                probs = counts / flat.numel()
                for i in range(self.vocabulary_size):
                    pp = expected[i].item()
                    sigma = (pp * (1.0 - pp) / flat.numel()) ** 0.5
                    self.assertLessEqual(abs(probs[i].item() - pp), 5.0 * sigma + 1e-12)
                joint = ((first[:, 0] == 2) & (first[:, 1] == 2)).float().mean().item()
                joint_expected = expected[2].item() ** 2
                sigma_j = (joint_expected * (1.0 - joint_expected) / N) ** 0.5
                self.assertLessEqual(abs(joint - joint_expected), 5.0 * sigma_j + 1e-12)


if __name__ == "__main__":
    unittest.main()
