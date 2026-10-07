"""An example's scalar schedule acts on all its features, not other examples' columns."""
import math

import unittest
from itertools import product

import torch

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import (
    CondOTScheduler,
    CosineScheduler,
    ScheduleTransformedModel,
    VPScheduler,
)


def coefficients(kind, time):
    if kind == "linear":
        return time, 1 - time, torch.ones_like(time), -torch.ones_like(time)
    if kind == "cosine":
        angle = math.pi / 2 * time
        return (
            angle.sin(),
            angle.cos(),
            math.pi / 2 * angle.cos(),
            -math.pi / 2 * angle.sin(),
        )
    deficit = 1 - time
    integral = 0.5 * deficit.square() * 19.9 + deficit * 0.1
    derivative = -deficit * 19.9 - 0.1
    alpha = (-0.5 * integral).exp()
    sigma = (-torch.expm1(-integral)).sqrt()
    return (
        alpha,
        sigma,
        -0.5 * derivative * alpha,
        0.5 * derivative * (-integral).exp() / sigma,
    )


SCHEDULERS = {"linear": CondOTScheduler, "cosine": CosineScheduler, "vp": VPScheduler}
CONVERSIONS = {
    "target_to_velocity": ("target", "velocity"),
    "epsilon_to_velocity": ("epsilon", "velocity"),
    "velocity_to_target": ("velocity", "target"),
    "epsilon_to_target": ("epsilon", "target"),
    "velocity_to_epsilon": ("velocity", "epsilon"),
    "target_to_epsilon": ("target", "epsilon"),
}


def analytic_representations(kind, x0, x1, time):
    alpha, sigma, da, ds = coefficients(kind, time)
    suffix = (None,) * (x0.ndim - time.ndim)
    alpha, sigma, da, ds = [value[(...,) + suffix] for value in (alpha, sigma, da, ds)]
    return {
        "epsilon": x0,
        "target": x1,
        "velocity": ds * x0 + da * x1,
        "state": sigma * x0 + alpha * x1,
    }


def _case_test_per_example_time_conversions_preserve_values_and_gradients(
    kind, shape, method
):
    torch.manual_seed(479)
    x0 = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x1 = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    time = torch.linspace(0.2, 0.7, shape[0], dtype=torch.float64).requires_grad_()
    representations = analytic_representations(kind, x0, x1, time)
    source, destination = CONVERSIONS[method]
    actual = getattr(AffineProbPath(SCHEDULERS[kind]()), method)(
        representations[source], representations["state"], time
    )
    expected = representations[destination]
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    weights = torch.linspace(-0.7, 0.9, actual.numel(), dtype=torch.float64).reshape_as(
        actual
    )
    targets = (x0, x1, time)
    actual_gradients = torch.autograd.grad(
        (actual * weights).sum(), targets, retain_graph=True, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        (expected * weights).sum(), targets, allow_unused=True
    )
    for target, observed, reference in zip(
        targets, actual_gradients, expected_gradients
    ):
        observed = torch.zeros_like(target) if observed is None else observed
        reference = torch.zeros_like(target) if reference is None else reference
        torch.testing.assert_close(observed, reference, rtol=1e-9, atol=1e-9)


def _case_test_scalar_and_elementwise_time_controls(time_shape, method):
    x0 = torch.linspace(-0.5, 0.3, 6, dtype=torch.float64).reshape(2, 3)
    x1 = x0 + 0.7
    time = torch.full(time_shape, 0.4, dtype=torch.float64)
    values = analytic_representations("cosine", x0, x1, time)
    source, destination = CONVERSIONS[method]
    actual = getattr(AffineProbPath(CosineScheduler()), method)(
        values[source], values["state"], time
    )
    torch.testing.assert_close(actual, values[destination], rtol=1e-11, atol=1e-11)


class ConditionalLinearVelocity(torch.nn.Module):
    def forward(self, x, t, endpoint, **extras):
        # Conditional velocity for X_t = (1-t) X_0 + t X_1.
        return (endpoint - x) / (1 - t).reshape(t.shape[0], *([1] * (x.ndim - 1)))


def _case_test_schedule_transformation_matches_new_conditional_velocity(shape, kind):
    x0 = (
        torch.linspace(-0.8, 0.6, math.prod(shape), dtype=torch.float64)
        .reshape(shape)
        .requires_grad_()
    )
    x1 = (x0.detach() * -0.3 + 0.4).requires_grad_()
    time = torch.linspace(0.25, 0.7, shape[0], dtype=torch.float64).requires_grad_()
    values = analytic_representations(kind, x0, x1, time)
    transformed = ScheduleTransformedModel(
        ConditionalLinearVelocity(), CondOTScheduler(), SCHEDULERS[kind]()
    )
    actual = transformed(x=values["state"], t=time, endpoint=x1)
    expected = values["velocity"]
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    actual_gradients = torch.autograd.grad(
        actual.square().sum(), (x0, x1, time), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected.square().sum(), (x0, x1, time))
    for observed, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(observed, reference, rtol=1e-9, atol=1e-9)


class TestAffineBatchCoefficients(unittest.TestCase):
    """Analytic parameter cases collected by the repository unittest workflow."""


def _install_parameter_cases(case_class, function, names, parameter_values):
    for index, values in enumerate(product(*parameter_values)):
        arguments = dict(zip(names, values))

        def run(self, function=function, arguments=arguments):
            function(**arguments)

        setattr(case_class, f"test_{function.__name__}_{index:03d}", run)


_install_parameter_cases(
    TestAffineBatchCoefficients,
    _case_test_per_example_time_conversions_preserve_values_and_gradients,
    ["kind", "shape", "method"],
    [["linear", "cosine", "vp"], [(3, 3), (2, 5), (2, 2, 2, 3)], list(CONVERSIONS)],
)
_install_parameter_cases(
    TestAffineBatchCoefficients,
    _case_test_scalar_and_elementwise_time_controls,
    ["time_shape", "method"],
    [[(), (2, 3)], list(CONVERSIONS)],
)
_install_parameter_cases(
    TestAffineBatchCoefficients,
    _case_test_schedule_transformation_matches_new_conditional_velocity,
    ["shape", "kind"],
    [[(3, 3), (2, 5), (2, 2, 2, 3)], ["cosine", "vp"]],
)
