"""A fixed-step integration interval retains its requested starting time."""
import unittest
from itertools import product

import torch

from flow_matching.solver import RiemannianODESolver
from flow_matching.utils.manifolds import FlatTorus


class ConstantVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor([0.1, 0.07], dtype=torch.float64))

    def forward(self, x, t, **extras):
        return self.rate.expand_as(x)


def _case_test_shifted_interval_matches_constant_velocity_solution_and_gradients(
    origin, method, project, intermediates
):
    model = ConstantVelocity()
    x0 = torch.tensor([[0.2, 0.3], [0.4, 0.5]], dtype=torch.float64, requires_grad=True)
    times = origin + torch.tensor([0.0, 0.125, 0.625, 1.0], dtype=torch.float64)
    actual = RiemannianODESolver(FlatTorus(), model).sample(
        x_init=x0,
        step_size=0.125,
        method=method,
        time_grid=times,
        projx=project,
        proju=project,
        return_intermediates=intermediates,
        enable_grad=True,
    )
    expected = x0.unsqueeze(0) + (times - origin)[:, None, None] * model.rate
    if not intermediates:
        expected = expected[-1]
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    weights = torch.linspace(-0.7, 0.9, actual.numel(), dtype=torch.float64).reshape_as(
        actual
    )
    actual_gradients = torch.autograd.grad((actual * weights).sum(), (x0, model.rate))
    expected_gradients = torch.autograd.grad(
        (expected * weights).sum(), (x0, model.rate)
    )
    for observed, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(observed, reference, rtol=1e-10, atol=1e-10)


def _case_test_zero_origin_control(method):
    model = ConstantVelocity()
    x0 = torch.tensor([[0.2, 0.3]], dtype=torch.float64)
    actual = RiemannianODESolver(FlatTorus(), model).sample(x0, 0.125, method=method)
    torch.testing.assert_close(actual, x0 + model.rate)


class TestRiemannianTimeOrigin(unittest.TestCase):
    """Analytic parameter cases collected by the repository unittest workflow."""


def _install_parameter_cases(case_class, function, names, parameter_values):
    for index, values in enumerate(product(*parameter_values)):
        arguments = dict(zip(names, values))

        def run(self, function=function, arguments=arguments):
            function(**arguments)

        setattr(case_class, f"test_{function.__name__}_{index:03d}", run)


_install_parameter_cases(
    TestRiemannianTimeOrigin,
    _case_test_shifted_interval_matches_constant_velocity_solution_and_gradients,
    ["origin", "method", "project", "intermediates"],
    [[-0.5, 0.5, 2.0], ["euler", "midpoint", "rk4"], [False, True], [False, True]],
)
_install_parameter_cases(
    TestRiemannianTimeOrigin,
    _case_test_zero_origin_control,
    ["method"],
    [["euler", "midpoint", "rk4"]],
)
