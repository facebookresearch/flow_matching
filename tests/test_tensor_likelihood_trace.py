"""The exact divergence is a trace over every non-batch coordinate."""
import math

import unittest
from itertools import product

import torch

from flow_matching.solver import ODESolver


class DiagonalFlow(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.rates = torch.nn.Parameter(
            torch.linspace(-0.2, 0.3, math.prod(shape), dtype=torch.float64).reshape(
                shape
            )
        )

    def forward(self, x, t, **extras):
        return x * self.rates


def gaussian_log_density(x):
    flat = x.flatten(1)
    return -0.5 * (flat.square().sum(1) + flat.shape[1] * math.log(2 * math.pi))


def _case_test_linear_pushforward_density_and_gradients_for_tensor_states(
    shape, exact, enable_grad
):
    torch.manual_seed(913)
    model = DiagonalFlow(shape[1:])
    x1 = (
        torch.linspace(-0.7, 0.9, math.prod(shape), dtype=torch.float64)
        .reshape(shape)
        .requires_grad_()
    )
    source, likelihood = ODESolver(model).compute_likelihood(
        x_1=x1,
        log_p0=gaussian_log_density,
        step_size=None,
        method="dopri5",
        time_grid=torch.tensor([1.0, 0.0], dtype=torch.float64),
        exact_divergence=exact,
        enable_grad=enable_grad,
        rtol=1e-10,
        atol=1e-12,
    )
    # The Gaussian pushforward under x1 = exp(rates) * x0 is available in closed form.
    expected_source = x1 * torch.exp(-model.rates)
    expected_likelihood = gaussian_log_density(expected_source) - model.rates.sum()
    torch.testing.assert_close(source, expected_source, rtol=2e-8, atol=2e-10)
    torch.testing.assert_close(likelihood, expected_likelihood, rtol=2e-8, atol=2e-10)
    if enable_grad:
        weights = torch.linspace(0.4, 1.0, shape[0], dtype=torch.float64)
        actual_gradients = torch.autograd.grad(
            (likelihood * weights).sum(), (x1, model.rates)
        )
        expected_gradients = torch.autograd.grad(
            (expected_likelihood * weights).sum(), (x1, model.rates)
        )
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected, rtol=3e-7, atol=3e-9)
    else:
        assert not source.requires_grad and not likelihood.requires_grad


class TestTensorLikelihoodTrace(unittest.TestCase):
    """Analytic parameter cases collected by the repository unittest workflow."""


def _install_parameter_cases(case_class, function, names, parameter_values):
    for index, values in enumerate(product(*parameter_values)):
        arguments = dict(zip(names, values))

        def run(self, function=function, arguments=arguments):
            function(**arguments)

        setattr(case_class, f"test_{function.__name__}_{index:03d}", run)


_install_parameter_cases(
    TestTensorLikelihoodTrace,
    _case_test_linear_pushforward_density_and_gradients_for_tensor_states,
    ["shape", "exact", "enable_grad"],
    [[(2, 3), (2, 2, 3), (2, 2, 2, 2), (3, 1, 2, 2)], [True, False], [True, False]],
)
