# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import math
import unittest

import torch
from flow_matching.path import (
    AffineProbPath,
    CondOTProbPath,
    GeodesicProbPath,
    MixtureDiscreteProbPath,
)
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.utils.manifolds import FlatTorus, Sphere


class TestAffineProbPath(unittest.TestCase):
    def test_affine_prob_path_sample(self):
        scheduler = CondOTScheduler()
        affine_prob_path = AffineProbPath(scheduler)
        x_0 = torch.randn(10, 5)
        x_1 = torch.randn(10, 5)
        t = torch.randn(10)
        sample = affine_prob_path.sample(x_0, x_1, t)
        self.assertEqual(sample.x_t.shape, x_0.shape)
        self.assertEqual(sample.dx_t.shape, x_0.shape)
        self.assertTrue((sample.t == t).all())
        self.assertTrue((sample.x_0 == x_0).all())
        self.assertTrue((sample.x_1 == x_1).all())

    def test_assert_sample_shape(self):
        scheduler = CondOTScheduler()
        path = AffineProbPath(scheduler)
        x_0 = torch.randn(10, 5)
        x_1 = torch.randn(10, 5)
        t = torch.randn(10)
        path.assert_sample_shape(x_0, x_1, t)

        x_0 = torch.randn(10, 5)
        x_1 = torch.randn(10, 5)
        t = torch.randn(5)
        with self.assertRaises(AssertionError):
            path.assert_sample_shape(x_0, x_1, t)

    def test_cond_ot_prob_path_sample(self):
        cond_ot_prob_path = CondOTProbPath()
        scheduler = CondOTScheduler()
        affine_path = AffineProbPath(scheduler)
        x_0 = torch.randn(10, 5)
        x_1 = torch.randn(10, 5)
        t = torch.randn(10)
        sample1 = cond_ot_prob_path.sample(x_0, x_1, t)
        sample2 = affine_path.sample(x_0, x_1, t)
        self.assertTrue(torch.allclose(sample1.x_t, sample2.x_t))

    def test_to_velocity(self):
        path = CondOTProbPath()
        x_1 = torch.randn(10, 5, dtype=torch.float64)
        x_t = torch.randn(10, 5, dtype=torch.float64)
        t = torch.randn(10, 5, dtype=torch.float64)
        velocity = path.target_to_velocity(x_1, x_t, t)
        target = path.velocity_to_target(velocity, x_t, t)
        self.assertTrue(torch.allclose(target, x_1))

    def test_to_epsilon(self):
        path = CondOTProbPath()
        x_1 = torch.randn(10, 5, dtype=torch.float64)
        x_t = torch.randn(10, 5, dtype=torch.float64)
        t = torch.randn(10, 5, dtype=torch.float64)
        epsilon = path.target_to_epsilon(x_1, x_t, t)
        target = path.epsilon_to_target(epsilon, x_t, t)
        self.assertTrue(torch.allclose(target, x_1))

    def test_epsilson_velocity(self):
        path = CondOTProbPath()
        velocity = torch.randn(10, 5, dtype=torch.float64)
        x_t = torch.randn(10, 5, dtype=torch.float64)
        t = torch.randn(10, 5, dtype=torch.float64)

        epsilon = path.velocity_to_epsilon(velocity, x_t, t)
        v = path.epsilon_to_velocity(epsilon, x_t, t)
        self.assertTrue(torch.allclose(v, velocity))


class TestGeodesicProbPath(unittest.TestCase):
    def test_sphere(self):
        manifold = Sphere()
        path = GeodesicProbPath(manifold=manifold, scheduler=CondOTScheduler())

        def wrap(samples):
            center = torch.cat(
                [torch.zeros_like(samples), torch.ones_like(samples[..., 0:1])], dim=-1
            )
            samples = (
                torch.cat([samples, torch.zeros_like(samples[..., 0:1])], dim=-1) / 2
            )
            return manifold.expmap(center, samples)

        x1 = manifold.projx(torch.rand(5, 5, dtype=torch.float64))
        x0 = torch.randn_like(x1)
        x0 = wrap(x0)
        x1 = wrap(x1)
        t = torch.rand(x0.size(0), dtype=torch.float64)

        sample = path.sample(t=t, x_0=x0, x_1=x1)

        # Check that x_t is on the sphere
        self.assertTrue(
            torch.allclose(
                sample.x_t.norm(2, -1), torch.ones(x0.size(0), dtype=torch.float64)
            )
        )

    def test_torus(self):
        manifold = FlatTorus()
        path = GeodesicProbPath(manifold=manifold, scheduler=CondOTScheduler())

        def wrap(samples):
            center = torch.zeros_like(samples)
            return manifold.expmap(center, samples)

        batch_size = 5
        coord1 = torch.rand(batch_size, dtype=torch.float64) * 4 - 2
        coord2_ = (
            torch.rand(batch_size, dtype=torch.float64)
            - torch.randint(high=2, size=(batch_size,), dtype=torch.float64) * 2
        )
        coord2 = coord2_ + (torch.floor(coord1) % 2)

        x1 = torch.stack([coord1, coord2], dim=1)
        x0 = torch.randn_like(x1)
        x0 = wrap(x0)
        x1 = wrap(x1)
        t = torch.rand(x0.size(0), dtype=torch.float64)

        sample = path.sample(t=t, x_0=x0, x_1=x1)

        self.assertTrue((sample.x_t < 2 * math.pi).all())


class TestMixtureDiscreteProbPath(unittest.TestCase):
    def test_mixture_discrete_prob_path_sample(self):
        scheduler = CondOTScheduler()
        discrete_prob_path = MixtureDiscreteProbPath(scheduler)
        x_0 = torch.randn(10, 5)
        x_1 = torch.randn(10, 5)
        t = torch.randn(10)
        sample = discrete_prob_path.sample(x_0, x_1, t)
        self.assertEqual(sample.x_t.shape, x_0.shape)
        self.assertTrue((sample.t == t).all())
        self.assertTrue((sample.x_0 == x_0).all())
        self.assertTrue((sample.x_1 == x_1).all())

        # Test at t=0
        t = torch.zeros(10)
        sample = discrete_prob_path.sample(x_0, x_1, t)
        self.assertTrue(torch.allclose(sample.x_t, x_0))
        # Test at t=1
        t = torch.ones(10)
        sample = discrete_prob_path.sample(x_0, x_1, t)
        self.assertTrue(torch.allclose(sample.x_t, x_1))

    def test_posterior_to_velocity(self):
        scheduler = CondOTScheduler()
        discrete_prob_path = MixtureDiscreteProbPath(scheduler)
        posterior_logits = torch.randn(10, 5)
        x_t = torch.randint(0, 5, size=[10])
        t = torch.randn(10)
        x_t_one_hot = torch.nn.functional.one_hot(x_t, num_classes=5)
        velocity = discrete_prob_path.posterior_to_velocity(posterior_logits, x_t, t)
        expected_velocity = (torch.softmax(posterior_logits, dim=-1) - x_t_one_hot) / (
            1 - t
        ).unsqueeze(-1)
        self.assertTrue(torch.allclose(velocity, expected_velocity))


if __name__ == "__main__":
    unittest.main()
