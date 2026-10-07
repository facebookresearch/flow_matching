"""VP coefficients and their inverse remain conditioned near the data endpoint."""
import math

import unittest
from itertools import product

import torch

from flow_matching.path.scheduler import VPScheduler


def _case_test_near_endpoint_noise_and_derivatives_use_nonzero_variance(
    dtype, betas, offset
):
    time = torch.tensor(
        1 - offset * torch.finfo(dtype).eps / 2, dtype=dtype, requires_grad=True
    )
    value = time.item()
    b, maximum = betas
    integral = 0.5 * (1 - value) ** 2 * (maximum - b) + (1 - value) * b
    integral_derivative = -(1 - value) * (maximum - b) - b
    sigma = math.sqrt(-math.expm1(-integral))
    expected = [
        math.exp(-integral / 2),
        sigma,
        -0.5 * integral_derivative * math.exp(-integral / 2),
        0.5 * integral_derivative * math.exp(-integral) / sigma,
    ]
    result = VPScheduler(beta_min=b, beta_max=maximum)(time)
    tolerance = 3e-6 if dtype == torch.float32 else 3e-13
    for actual, reference in zip(
        (result.alpha_t, result.sigma_t, result.d_alpha_t, result.d_sigma_t), expected
    ):
        assert torch.isfinite(actual) and actual != 0
        torch.testing.assert_close(
            actual, torch.tensor(reference, dtype=dtype), rtol=tolerance, atol=0.0
        )
    (sigma_derivative,) = torch.autograd.grad(result.sigma_t, (time,))
    torch.testing.assert_close(
        sigma_derivative,
        torch.tensor(expected[-1], dtype=dtype),
        rtol=tolerance,
        atol=0.0,
    )


def _case_test_inverse_matches_independent_integrated_rate_root_and_gradient(
    dtype, betas, snr_value
):
    b, maximum = betas
    snr = torch.tensor(snr_value, dtype=dtype, requires_grad=True)
    integral = math.log1p((1 / snr.item()) ** 2)
    deficit = 2 * integral / (b + math.sqrt(b * b + 2 * (maximum - b) * integral))
    expected = 1 - deficit
    actual = VPScheduler(beta_min=b, beta_max=maximum).snr_inverse(snr)
    assert torch.isfinite(actual)
    tolerance = 5e-6 if dtype == torch.float32 else 5e-13
    torch.testing.assert_close(
        actual,
        torch.tensor(expected, dtype=dtype),
        rtol=0.0,
        atol=4 * torch.finfo(dtype).eps,
    )
    (derivative,) = torch.autograd.grad(actual, (snr,))
    # Differentiate the integral constraint b*q + .5*(B-b)*q^2 = log(1+snr^-2).
    expected_derivative = 2 / (
        snr.item() * (snr.item() ** 2 + 1) * (b + (maximum - b) * deficit)
    )
    assert torch.isfinite(derivative)
    torch.testing.assert_close(
        derivative,
        torch.tensor(expected_derivative, dtype=dtype),
        rtol=tolerance,
        atol=torch.finfo(dtype).tiny,
    )


def _case_test_regular_time_round_trip_control(dtype, betas):
    scheduler = VPScheduler(beta_min=betas[0], beta_max=betas[1])
    times = torch.tensor([0.1, 0.3, 0.7, 0.9], dtype=dtype)
    result = scheduler(times)
    torch.testing.assert_close(
        scheduler.snr_inverse(result.alpha_t / result.sigma_t),
        times,
        rtol=3e-6,
        atol=3e-6,
    )


class TestVpSchedulerConditioning(unittest.TestCase):
    """Analytic parameter cases collected by the repository unittest workflow."""


def _install_parameter_cases(case_class, function, names, parameter_values):
    for index, values in enumerate(product(*parameter_values)):
        arguments = dict(zip(names, values))

        def run(self, function=function, arguments=arguments):
            function(**arguments)

        setattr(case_class, f"test_{function.__name__}_{index:03d}", run)


_install_parameter_cases(
    TestVpSchedulerConditioning,
    _case_test_near_endpoint_noise_and_derivatives_use_nonzero_variance,
    ["dtype", "betas", "offset"],
    [[torch.float32, torch.float64], [(0.1, 20.0), (0.4, 0.4)], [1, 4, 32, 1024]],
)
_install_parameter_cases(
    TestVpSchedulerConditioning,
    _case_test_inverse_matches_independent_integrated_rate_root_and_gradient,
    ["dtype", "betas", "snr_value"],
    [
        [torch.float32, torch.float64],
        [(0.1, 20.0), (0.4, 0.4)],
        [1.0, 10.0, 100.0, 1e5, 1e20],
    ],
)
_install_parameter_cases(
    TestVpSchedulerConditioning,
    _case_test_regular_time_round_trip_control,
    ["dtype", "betas"],
    [[torch.float32, torch.float64], [(0.1, 20.0), (0.4, 0.4)]],
)
