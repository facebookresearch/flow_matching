"""A zero jump contribution does not evaluate an impossible target's log probability."""
import math

import unittest
from itertools import product

import torch

from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler


def scalar_loss_and_gradient(logits, target, current, time):
    finite = [value for value in logits if math.isfinite(value)]
    maximum = max(finite)
    exponentials = [
        0.0 if value == -math.inf else math.exp(value - maximum) for value in logits
    ]
    probabilities = [value / sum(exponentials) for value in exponentials]
    coefficient = 2 * time / (1 - time * time)
    aligned = target == current
    if aligned:
        loss = coefficient * (1 - probabilities[current])
    else:
        loss = -coefficient * (probabilities[current] + math.log(probabilities[target]))
    gradients = []
    for j, probability in enumerate(probabilities):
        probability_gradient = probabilities[current] * ((j == current) - probability)
        log_gradient = 0.0 if aligned else (j == target) - probability
        gradients.append(-coefficient * (probability_gradient + log_gradient))
    return loss, gradients


def _case_test_masked_aligned_targets_match_piecewise_loss_and_logit_jacobian(
    dtype, reduction, shape, pattern
):
    count = math.prod(shape)
    data = torch.linspace(-0.8, 0.7, count * 4, dtype=dtype).reshape(*shape, 4)
    target = torch.arange(count).reshape(shape) % 4
    current = target.clone()
    if pattern == "mixed":
        current.flatten()[1::2] = (target.flatten()[1::2] + 1) % 4
    if pattern != "finite_control":
        aligned = current == target
        for index in range(count):
            if aligned.flatten()[index]:
                data.reshape(count, 4)[index, target.flatten()[index]] = -torch.inf
    logits = data.clone().requires_grad_()
    times = torch.linspace(0.2, 0.7, shape[0], dtype=dtype)
    expected_values, expected_gradients = [], []
    for index, row in enumerate(data.reshape(count, 4).tolist()):
        value, gradient = scalar_loss_and_gradient(
            row,
            target.flatten()[index].item(),
            current.flatten()[index].item(),
            times[index // shape[1]].item(),
        )
        expected_values.append(value)
        expected_gradients.append(gradient)
    expected = torch.tensor(expected_values, dtype=dtype).reshape(shape)
    reference_gradient = torch.tensor(expected_gradients, dtype=dtype).reshape_as(
        logits
    )
    if reduction == "mean":
        expected = expected.mean()
        reference_gradient = reference_gradient / count
    elif reduction == "sum":
        expected = expected.sum()
    actual = MixturePathGeneralizedKL(
        MixtureDiscreteProbPath(PolynomialConvexScheduler(2)), reduction=reduction
    )(logits, target, current, times)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-11
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    (actual_gradient,) = torch.autograd.grad(actual.sum(), (logits,))
    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(
        actual_gradient, reference_gradient, rtol=tolerance, atol=tolerance
    )


def _case_test_impossible_unaligned_target_retains_infinite_loss(dtype, reduction):
    logits = torch.tensor([[[-torch.inf, 0.3, -0.2]]], dtype=dtype)
    actual = MixturePathGeneralizedKL(
        MixtureDiscreteProbPath(PolynomialConvexScheduler(2)), reduction=reduction
    )(
        logits,
        torch.tensor([[0]]),
        torch.tensor([[1]]),
        torch.tensor([0.4], dtype=dtype),
    )
    assert torch.isposinf(actual).all()


class TestGeneralizedKlMaskedSupport(unittest.TestCase):
    """Analytic parameter cases collected by the repository unittest workflow."""


def _install_parameter_cases(case_class, function, names, parameter_values):
    for index, values in enumerate(product(*parameter_values)):
        arguments = dict(zip(names, values))

        def run(self, function=function, arguments=arguments):
            function(**arguments)

        setattr(case_class, f"test_{function.__name__}_{index:03d}", run)


_install_parameter_cases(
    TestGeneralizedKlMaskedSupport,
    _case_test_masked_aligned_targets_match_piecewise_loss_and_logit_jacobian,
    ["dtype", "reduction", "shape", "pattern"],
    [
        [torch.float32, torch.float64],
        ["none", "mean", "sum"],
        [(2, 3), (3, 2)],
        ["masked_aligned", "mixed", "finite_control"],
    ],
)
_install_parameter_cases(
    TestGeneralizedKlMaskedSupport,
    _case_test_impossible_unaligned_target_retains_infinite_loss,
    ["dtype", "reduction"],
    [[torch.float32, torch.float64], ["none", "mean", "sum"]],
)
