"""Requested nearest-grid snapshots retain duplicates and the caller's order."""
import unittest
from itertools import product

import torch

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver


class TimeVaryingPosterior(torch.nn.Module):
    def forward(self, x, t, **extras):
        labels = torch.arange(4, dtype=torch.float64)
        logits = torch.cos(labels + 5 * t.reshape(-1, *([1] * x.ndim)))
        logits = logits.expand(*x.shape, 4) + 0.3 * torch.nn.functional.one_hot(x, 4)
        return logits.softmax(-1)


def _case_test_nearest_grid_queries_match_full_trajectory_snapshots(
    seed, shape, queries
):
    solver = MixtureDiscreteEulerSolver(
        TimeVaryingPosterior(), MixtureDiscreteProbPath(CondOTScheduler()), 4
    )
    x0 = torch.arange(torch.tensor(shape).prod().item()).reshape(shape) % 4
    integration_grid = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    requested = torch.tensor(queries)
    torch.manual_seed(seed)
    # All aligned checkpoints give the reference trajectory with the same model calls and RNG draws.
    full = solver.sample(
        x0,
        0.25,
        time_grid=integration_grid,
        return_intermediates=True,
        dtype_categorical=torch.float64,
    )
    expected_indices = torch.argmin(
        (requested[:, None] - integration_grid[None, :]).abs(), dim=1
    )
    expected = full[expected_indices]
    torch.manual_seed(seed)
    actual = solver.sample(
        x0,
        0.25,
        time_grid=requested,
        return_intermediates=True,
        dtype_categorical=torch.float64,
    )
    assert actual.shape == (len(queries), *shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _case_test_aligned_grid_and_final_only_controls(seed):
    solver = MixtureDiscreteEulerSolver(
        TimeVaryingPosterior(), MixtureDiscreteProbPath(CondOTScheduler()), 4
    )
    x0 = torch.tensor([[0, 1, 2], [3, 1, 0]])
    times = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    torch.manual_seed(seed)
    full = solver.sample(x0, 0.25, time_grid=times, return_intermediates=True)
    torch.manual_seed(seed)
    final = solver.sample(x0, 0.25, time_grid=times)
    torch.testing.assert_close(final, full[-1], rtol=0, atol=0)


class TestDiscreteIntermediateQueries(unittest.TestCase):
    """Analytic parameter cases collected by the repository unittest workflow."""


def _install_parameter_cases(case_class, function, names, parameter_values):
    for index, values in enumerate(product(*parameter_values)):
        arguments = dict(zip(names, values))

        def run(self, function=function, arguments=arguments):
            function(**arguments)

        setattr(case_class, f"test_{function.__name__}_{index:03d}", run)


_install_parameter_cases(
    TestDiscreteIntermediateQueries,
    _case_test_nearest_grid_queries_match_full_trajectory_snapshots,
    ["seed", "shape", "queries"],
    [
        [19, 203, 701],
        [(2, 3), (3, 2, 2)],
        [
            [0.0, 0.01, 0.04, 0.24, 0.26, 0.74, 0.76, 0.99, 1.0],
            [0.0, 0.75, 0.25, 0.5, 1.0],
            [0.0, 0.25, 0.25, 0.5, 0.75, 1.0],
        ],
    ],
)
_install_parameter_cases(
    TestDiscreteIntermediateQueries,
    _case_test_aligned_grid_and_final_only_controls,
    ["seed"],
    [[19, 203]],
)


class RecordingPosterior(TimeVaryingPosterior):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x, t, **extras):
        self.inputs.append(x.clone())
        return super().forward(x, t, **extras)


def _case_test_explicit_nonuniform_grid_and_duplicate_time_controls(seed, shape, times):
    model = RecordingPosterior()
    solver = MixtureDiscreteEulerSolver(
        model, MixtureDiscreteProbPath(CondOTScheduler()), 4
    )
    x0 = torch.arange(torch.tensor(shape).prod().item()).reshape(shape) % 4
    grid = torch.tensor(times)
    torch.manual_seed(seed)
    # Inputs seen by the public model are the states at each step's starting
    # checkpoint. Final-only sampling supplies the terminal checkpoint and is
    # independent of the intermediate-result storage under test.
    final = solver.sample(x0, None, time_grid=grid, dtype_categorical=torch.float64)
    assert len(model.inputs) == len(times) - 1
    expected = torch.stack(model.inputs + [final])
    model.inputs.clear()
    torch.manual_seed(seed)
    actual = solver.sample(
        x0,
        None,
        time_grid=grid,
        return_intermediates=True,
        dtype_categorical=torch.float64,
    )
    assert len(model.inputs) == len(times) - 1
    assert actual.shape == (len(times), *shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], x0, rtol=0, atol=0)
    torch.testing.assert_close(actual[-1], final, rtol=0, atol=0)
    for index in range(1, len(times)):
        if times[index] == times[index - 1]:
            torch.testing.assert_close(actual[index], actual[index - 1], rtol=0, atol=0)


_install_parameter_cases(
    TestDiscreteIntermediateQueries,
    _case_test_explicit_nonuniform_grid_and_duplicate_time_controls,
    ["seed", "shape", "times"],
    [
        [19, 203, 701],
        [(2, 3), (3, 2, 2)],
        [
            [0.0, 0.125, 0.375, 0.625, 1.0],
            [0.0, 0.125, 0.125, 0.375, 0.625, 1.0],
        ],
    ],
)
