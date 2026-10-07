# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch

from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.utils.manifolds import Euclidean, FlatTorus, Sphere


class TestGeodesicTensorEvents(unittest.TestCase):
    def test_euclidean_state_and_velocity(self):
        for shape in [(3, 4), (3, 5, 4), (3, 2, 5, 4), (3, 1, 2, 1, 4)]:
            for dtype in [torch.float32, torch.float64]:
                for n in [1, 2]:
                    with self.subTest(shape=shape, dtype=dtype, n=n):
                        x_0 = torch.linspace(
                            -1, 1, math.prod(shape), dtype=dtype
                        ).reshape(shape)
                        x_1 = x_0.flip(-1) + 0.5
                        t = torch.tensor([0.2, 0.5, 0.8], dtype=dtype)
                        path = GeodesicProbPath(
                            PolynomialConvexScheduler(n), Euclidean()
                        )
                        sample = path.sample(x_0, x_1, t)
                        alpha = (t**n).reshape(3, *([1] * (len(shape) - 1)))
                        rate = (n * t ** (n - 1)).reshape_as(alpha)
                        self.assert_sample(sample, x_0, x_1, t)
                        torch.testing.assert_close(
                            sample.x_t, x_0 + alpha * (x_1 - x_0)
                        )
                        torch.testing.assert_close(sample.dx_t, rate * (x_1 - x_0))

    def test_flat_torus_state_and_velocity(self):
        for shape in [(3, 4), (3, 5, 4), (3, 2, 5, 4)]:
            for dtype in [torch.float32, torch.float64]:
                for n in [1, 2]:
                    with self.subTest(shape=shape, dtype=dtype, n=n):
                        x_0 = torch.linspace(
                            6.1, 6.2, math.prod(shape), dtype=dtype
                        ).reshape(shape)
                        delta = torch.linspace(
                            0.4, 0.7, math.prod(shape), dtype=dtype
                        ).reshape(shape)
                        x_1 = (x_0 + delta) % (2 * math.pi)
                        t = torch.tensor([0.2, 0.5, 0.8], dtype=dtype)
                        path = GeodesicProbPath(
                            PolynomialConvexScheduler(n), FlatTorus()
                        )
                        sample = path.sample(x_0, x_1, t)
                        alpha = (t**n).reshape(3, *([1] * (len(shape) - 1)))
                        rate = (n * t ** (n - 1)).reshape_as(alpha)
                        self.assert_sample(sample, x_0, x_1, t)
                        torch.testing.assert_close(
                            sample.x_t, (x_0 + alpha * delta) % (2 * math.pi)
                        )
                        torch.testing.assert_close(sample.dx_t, rate * delta)

    def test_sphere_state_and_velocity(self):
        for shape in [(3,), (3, 5), (3, 2, 5)]:
            for dtype in [torch.float32, torch.float64]:
                for n in [1, 2]:
                    with self.subTest(shape=shape, dtype=dtype, n=n):
                        phi = torch.linspace(
                            -0.5, 0.5, math.prod(shape), dtype=dtype
                        ).reshape(shape)
                        theta = torch.linspace(
                            0.4, 1.2, math.prod(shape), dtype=dtype
                        ).reshape(shape)
                        t = torch.tensor([0.2, 0.5, 0.8], dtype=dtype)
                        x_0 = self.sphere_point(phi)
                        x_1 = self.sphere_point(phi + theta)
                        path = GeodesicProbPath(PolynomialConvexScheduler(n), Sphere())
                        sample = path.sample(x_0, x_1, t)
                        alpha = (t**n).reshape(3, *([1] * (len(shape) - 1)))
                        rate = (n * t ** (n - 1)).reshape_as(alpha)
                        angle = phi + alpha * theta
                        expected_velocity = torch.stack(
                            [
                                -torch.sin(angle),
                                torch.cos(angle),
                                torch.zeros_like(angle),
                            ],
                            dim=-1,
                        ) * (rate * theta).unsqueeze(-1)
                        self.assert_sample(sample, x_0, x_1, t)
                        torch.testing.assert_close(sample.x_t, self.sphere_point(angle))
                        torch.testing.assert_close(sample.dx_t, expected_velocity)

    def test_euclidean_and_torus_gradients(self):
        for shape in [(3, 4), (3, 5, 4), (3, 2, 5, 4)]:
            for manifold in [Euclidean(), FlatTorus()]:
                for n in [1, 2]:
                    with self.subTest(
                        shape=shape, manifold=type(manifold).__name__, n=n
                    ):
                        dtype = torch.float64
                        x_0 = (
                            torch.linspace(0.2, 0.5, math.prod(shape), dtype=dtype)
                            .reshape(shape)
                            .requires_grad_()
                        )
                        x_1 = (
                            torch.linspace(0.9, 1.4, math.prod(shape), dtype=dtype)
                            .reshape(shape)
                            .requires_grad_()
                        )
                        t = torch.tensor(
                            [0.2, 0.5, 0.8], dtype=dtype, requires_grad=True
                        )
                        path = GeodesicProbPath(PolynomialConvexScheduler(n), manifold)
                        sample = path.sample(x_0, x_1, t)
                        alpha = (t**n).reshape(3, *([1] * (len(shape) - 1)))
                        rate = (n * t ** (n - 1)).reshape_as(alpha)
                        expected_x = x_0 + alpha * (x_1 - x_0)
                        expected_v = rate * (x_1 - x_0)
                        weight = torch.linspace(
                            -0.8, 0.9, math.prod(shape), dtype=dtype
                        ).reshape(shape)
                        actual = (
                            sample.x_t * weight + sample.dx_t * weight.flip(-1)
                        ).sum()
                        expected = (
                            expected_x * weight + expected_v * weight.flip(-1)
                        ).sum()
                        actual_grads = torch.autograd.grad(
                            actual, (x_0, x_1, t), retain_graph=True
                        )
                        expected_grads = torch.autograd.grad(expected, (x_0, x_1, t))
                        for actual_grad, expected_grad in zip(
                            actual_grads, expected_grads
                        ):
                            torch.testing.assert_close(actual_grad, expected_grad)

    def test_sphere_gradients(self):
        for shape in [(3,), (3, 5), (3, 2, 5)]:
            for n in [1, 2]:
                with self.subTest(shape=shape, n=n):
                    dtype = torch.float64
                    phi = (
                        torch.linspace(-0.5, 0.5, math.prod(shape), dtype=dtype)
                        .reshape(shape)
                        .requires_grad_()
                    )
                    theta = (
                        torch.linspace(0.4, 1.2, math.prod(shape), dtype=dtype)
                        .reshape(shape)
                        .requires_grad_()
                    )
                    t = torch.tensor([0.2, 0.5, 0.8], dtype=dtype, requires_grad=True)
                    path = GeodesicProbPath(PolynomialConvexScheduler(n), Sphere())
                    sample = path.sample(
                        self.sphere_point(phi), self.sphere_point(phi + theta), t
                    )
                    alpha = (t**n).reshape(3, *([1] * (len(shape) - 1)))
                    rate = (n * t ** (n - 1)).reshape_as(alpha)
                    angle = phi + alpha * theta
                    expected_x = self.sphere_point(angle)
                    expected_v = torch.stack(
                        [-torch.sin(angle), torch.cos(angle), torch.zeros_like(angle)],
                        dim=-1,
                    ) * (rate * theta).unsqueeze(-1)
                    weight = torch.linspace(
                        -0.8, 0.9, expected_x.numel(), dtype=dtype
                    ).reshape_as(expected_x)
                    actual = (sample.x_t * weight + sample.dx_t * weight.flip(-1)).sum()
                    expected = (
                        expected_x * weight + expected_v * weight.flip(-1)
                    ).sum()
                    actual_grads = torch.autograd.grad(
                        actual, (phi, theta, t), retain_graph=True
                    )
                    expected_grads = torch.autograd.grad(expected, (phi, theta, t))
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad)

    def test_endpoints_and_time_validation(self):
        for shape in [(3, 5, 4), (3, 2, 5, 4)]:
            for manifold in [Euclidean(), FlatTorus()]:
                for n in [1, 2]:
                    with self.subTest(
                        shape=shape, manifold=type(manifold).__name__, n=n
                    ):
                        x_0 = torch.linspace(
                            0.2, 0.5, math.prod(shape), dtype=torch.float64
                        ).reshape(shape)
                        x_1 = x_0 + 0.7
                        path = GeodesicProbPath(PolynomialConvexScheduler(n), manifold)
                        for time, expected in [(0.0, x_0), (1.0, x_1)]:
                            sample = path.sample(
                                x_0, x_1, torch.full((3,), time, dtype=torch.float64)
                            )
                            torch.testing.assert_close(sample.x_t, expected)
                        with self.assertRaises(AssertionError):
                            path.sample(x_0, x_1, torch.ones(3, 1))
                        with self.assertRaises(AssertionError):
                            path.sample(x_0, x_1, torch.ones(4))

    def assert_sample(self, sample, x_0, x_1, t):
        self.assertEqual(sample.x_t.shape, x_0.shape)
        self.assertEqual(sample.dx_t.shape, x_0.shape)
        self.assertEqual(sample.x_t.dtype, x_0.dtype)
        self.assertEqual(sample.dx_t.dtype, x_0.dtype)
        self.assertIs(sample.x_0, x_0)
        self.assertIs(sample.x_1, x_1)
        torch.testing.assert_close(sample.t, t[:, None])

    @staticmethod
    def sphere_point(angle):
        return torch.stack(
            [torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)], dim=-1
        )


if __name__ == "__main__":
    unittest.main()
