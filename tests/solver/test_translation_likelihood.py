# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch

from flow_matching.solver import ODESolver
from flow_matching.utils import gradient


class Translation(torch.nn.Module):
    def __init__(
        self, time_dependent, trainable=True, connected=False, dtype=torch.float32
    ):
        super().__init__()
        bias = torch.tensor([0.2, -0.3, 0.4, -0.1], dtype=dtype)
        rate = torch.tensor([-0.1, 0.2, 0.3, -0.2], dtype=dtype)
        if trainable:
            self.bias = torch.nn.Parameter(bias)
            self.rate = torch.nn.Parameter(rate)
        else:
            self.register_buffer("bias", bias)
            self.register_buffer("rate", rate)
        self.time_dependent = time_dependent
        self.connected = connected
        self.times = []

    def forward(self, x, t, offset):
        self.times.append(t.detach().clone())
        velocity = self.bias + offset
        if self.time_dependent:
            velocity = velocity + self.rate * t
        velocity = velocity.expand_as(x)
        return velocity + x * 0 if self.connected else velocity


def gaussian_log_density(x):
    return -0.5 * (x.square().sum(-1) + x.shape[-1] * math.log(2 * math.pi))


class TestTranslationLikelihood(unittest.TestCase):
    def test_translation_density_and_gradients(self):
        for time_dependent in [False, True]:
            methods = (
                ["midpoint", "rk4", "dopri5"]
                if time_dependent
                else ["euler", "midpoint", "rk4", "dopri5"]
            )
            for trainable in [False, True]:
                for exact in [False, True]:
                    for enable_grad in [False, True]:
                        for method in methods:
                            with self.subTest(
                                time_dependent=time_dependent,
                                trainable=trainable,
                                exact=exact,
                                enable_grad=enable_grad,
                                method=method,
                            ):
                                self.check_analytic_translation(
                                    time_dependent,
                                    trainable,
                                    exact,
                                    enable_grad,
                                    method,
                                    torch.float32,
                                )

    def test_double_precision_exact_translation(self):
        # The existing float64 Hutchinson projection bug belongs to PR #98.
        for time_dependent in [False, True]:
            for enable_grad in [False, True]:
                for method in ["rk4", "dopri5"]:
                    with self.subTest(
                        time_dependent=time_dependent,
                        enable_grad=enable_grad,
                        method=method,
                    ):
                        self.check_analytic_translation(
                            time_dependent,
                            True,
                            True,
                            enable_grad,
                            method,
                            torch.float64,
                        )

    def check_analytic_translation(
        self, time_dependent, trainable, exact, enable_grad, method, dtype
    ):
        model = Translation(time_dependent, trainable=trainable, dtype=dtype)
        x_1 = torch.linspace(-0.6, 0.9, 12, dtype=dtype).reshape(3, 4).requires_grad_()
        times = torch.tensor([1.0, 0.75, 0.25, 0.0], dtype=dtype)
        offset = 0.125
        torch.manual_seed(631)
        trajectory, likelihood = ODESolver(model).compute_likelihood(
            x_1=x_1,
            log_p0=gaussian_log_density,
            step_size=None if method == "dopri5" else 0.125,
            method=method,
            atol=1e-9 if dtype == torch.float64 else 1e-7,
            rtol=1e-9 if dtype == torch.float64 else 1e-7,
            time_grid=times,
            exact_divergence=exact,
            enable_grad=enable_grad,
            return_intermediates=True,
            offset=offset,
        )
        expected_trajectory = x_1.unsqueeze(0) + (times - times[0])[:, None, None] * (
            model.bias + offset
        )
        if time_dependent:
            expected_trajectory = (
                expected_trajectory
                + 0.5 * (times.square() - times[0].square())[:, None, None] * model.rate
            )
        expected_likelihood = gaussian_log_density(expected_trajectory[-1])
        tolerance = 3e-8 if dtype == torch.float64 else 3e-5
        torch.testing.assert_close(
            trajectory, expected_trajectory, rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(
            likelihood, expected_likelihood, rtol=tolerance, atol=tolerance
        )
        self.assertEqual(trajectory.dtype, dtype)
        if not enable_grad:
            self.assertFalse(trajectory.requires_grad)
            self.assertFalse(likelihood.requires_grad)
            return
        variables = [x_1]
        if trainable:
            variables.append(model.bias)
            if time_dependent:
                variables.append(model.rate)
        weights = torch.tensor([0.2, 0.6, 0.9], dtype=dtype)
        actual_gradients = torch.autograd.grad(
            (likelihood * weights).sum(), variables, create_graph=True
        )
        expected_gradients = torch.autograd.grad(
            (expected_likelihood * weights).sum(), variables, create_graph=True
        )
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        if trainable:
            actual_curvature = torch.autograd.grad(
                actual_gradients[1].sum(), model.bias
            )[0]
            expected_curvature = torch.autograd.grad(
                expected_gradients[1].sum(), model.bias
            )[0]
            torch.testing.assert_close(
                actual_curvature, expected_curvature, rtol=tolerance, atol=tolerance
            )

    def test_connected_field_compatibility(self):
        for time_dependent in [False, True]:
            for exact in [False, True]:
                for enable_grad in [False, True]:
                    for method in ["euler", "rk4", "dopri5"]:
                        with self.subTest(
                            time_dependent=time_dependent,
                            exact=exact,
                            enable_grad=enable_grad,
                            method=method,
                        ):
                            results = []
                            for connected in [False, True]:
                                model = Translation(time_dependent, connected=connected)
                                x_1 = (
                                    torch.linspace(-0.6, 0.9, 12)
                                    .reshape(3, 4)
                                    .requires_grad_()
                                )
                                torch.manual_seed(631)
                                source, likelihood = ODESolver(
                                    model
                                ).compute_likelihood(
                                    x_1=x_1,
                                    log_p0=gaussian_log_density,
                                    step_size=None if method == "dopri5" else 0.125,
                                    method=method,
                                    time_grid=torch.tensor([1.0, 0.75, 0.25, 0.0]),
                                    exact_divergence=exact,
                                    enable_grad=enable_grad,
                                    return_intermediates=True,
                                    offset=0.125,
                                )
                                variables = [x_1, model.bias]
                                if time_dependent:
                                    variables.append(model.rate)
                                gradients = (
                                    torch.autograd.grad(likelihood.sum(), variables)
                                    if enable_grad
                                    else []
                                )
                                results.append(
                                    (
                                        source,
                                        likelihood,
                                        gradients,
                                        model.times,
                                        torch.get_rng_state(),
                                    )
                                )
                            left, right = results
                            for actual, expected in zip(left[:2], right[:2]):
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                            for actual, expected in zip(left[2], right[2]):
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                            self.assertEqual(len(left[3]), len(right[3]))
                            for actual, expected in zip(left[3], right[3]):
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                            self.assertTrue(torch.equal(left[4], right[4]))

    def test_nonconstant_spatial_derivative_control(self):
        for exact in [False, True]:
            for method in ["rk4", "dopri5"]:
                with self.subTest(exact=exact, method=method):
                    rates = torch.nn.Parameter(torch.tensor([-0.3, 0.2, 0.4, -0.1]))
                    x_1 = torch.linspace(-0.6, 0.9, 12).reshape(3, 4).requires_grad_()
                    torch.manual_seed(631)
                    source, likelihood = ODESolver(
                        lambda x, t: x * rates
                    ).compute_likelihood(
                        x_1=x_1,
                        log_p0=gaussian_log_density,
                        step_size=None if method == "dopri5" else 0.125,
                        method=method,
                        atol=1e-7,
                        rtol=1e-7,
                        exact_divergence=exact,
                        enable_grad=True,
                    )
                    expected_source = x_1 * torch.exp(-rates)
                    expected_likelihood = (
                        gaussian_log_density(expected_source) - rates.sum()
                    )
                    torch.testing.assert_close(
                        source, expected_source, rtol=3e-5, atol=3e-5
                    )
                    torch.testing.assert_close(
                        likelihood, expected_likelihood, rtol=3e-5, atol=3e-5
                    )
                    actual_gradients = torch.autograd.grad(
                        likelihood.sum(), (x_1, rates)
                    )
                    expected_gradients = torch.autograd.grad(
                        expected_likelihood.sum(), (x_1, rates)
                    )
                    for actual, expected in zip(actual_gradients, expected_gradients):
                        torch.testing.assert_close(
                            actual, expected, rtol=3e-5, atol=3e-5
                        )

    def test_gradient_default_stays_strict(self):
        x = torch.ones(3, 4, requires_grad=True)
        for trainable in [False, True]:
            with self.subTest(trainable=trainable):
                output = Translation(False, trainable=trainable)(
                    x, torch.tensor(0.5), offset=0.125
                )
                with self.assertRaises(RuntimeError):
                    gradient(output, x)


if __name__ == "__main__":
    unittest.main()
