"""The zero-tangent exponential map has a finite differential."""
import unittest
from itertools import product

import torch

from flow_matching.utils.manifolds import Sphere


def _case_test_zero_and_small_tangents_match_retraction_values_and_gradients(
    dtype, shape, magnitude
):
    torch.manual_seed(297)
    sphere = Sphere()
    unit = torch.randn(shape, dtype=dtype)
    unit /= unit.norm(dim=-1, keepdim=True)
    x = unit.clone().requires_grad_()
    tangent = sphere.proju(unit, torch.randn(shape, dtype=dtype)) * magnitude
    u = tangent.clone().requires_grad_()
    actual = sphere.expmap(x, u)
    # The documented small-norm branch is normalized x+u.
    expected = torch.nn.functional.normalize(x + u, dim=-1)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    weights = torch.linspace(-0.9, 0.8, actual.numel(), dtype=dtype).reshape_as(actual)
    actual_gradients = torch.autograd.grad(
        (actual * weights).sum(), (x, u), retain_graph=True
    )
    expected_gradients = torch.autograd.grad((expected * weights).sum(), (x, u))
    for observed, reference in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference, rtol=tolerance, atol=tolerance)


def _case_test_nonzero_tangent_follows_great_circle_and_angle_derivative(
    dtype, angle_value
):
    angle = torch.tensor(angle_value, dtype=dtype, requires_grad=True)
    x = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype)
    direction = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype)
    actual = Sphere().expmap(x, angle * direction)
    expected = torch.stack((angle.cos(), angle.sin(), angle * 0)).unsqueeze(0)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    weights = torch.tensor([[0.3, -0.7, 0.8]], dtype=dtype)
    (observed,) = torch.autograd.grad((actual * weights).sum(), (angle,))
    reference = -0.3 * angle.sin() - 0.7 * angle.cos()
    torch.testing.assert_close(observed, reference, rtol=tolerance, atol=tolerance)


class TestSphereExpmapZeroTangent(unittest.TestCase):
    """Analytic parameter cases collected by the repository unittest workflow."""


def _install_parameter_cases(case_class, function, names, parameter_values):
    for index, values in enumerate(product(*parameter_values)):
        arguments = dict(zip(names, values))

        def run(self, function=function, arguments=arguments):
            function(**arguments)

        setattr(case_class, f"test_{function.__name__}_{index:03d}", run)


_install_parameter_cases(
    TestSphereExpmapZeroTangent,
    _case_test_zero_and_small_tangents_match_retraction_values_and_gradients,
    ["dtype", "shape", "magnitude"],
    [[torch.float32, torch.float64], [(2, 3), (2, 2, 4)], [0.0, 1e-9]],
)
_install_parameter_cases(
    TestSphereExpmapZeroTangent,
    _case_test_nonzero_tangent_follows_great_circle_and_angle_derivative,
    ["dtype", "angle_value"],
    [[torch.float32, torch.float64], [-0.6, 0.2, 1.1]],
)
