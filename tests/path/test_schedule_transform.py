# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from flow_matching.path.scheduler import (
    CondOTScheduler,
    CosineScheduler,
    ScheduleTransformedModel,
)
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper


class DummyModel(ModelWrapper):
    def __init__(self):
        super().__init__(None)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        return x * t**2


class TestScheduleTransformedModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10
        self.data_dim = 2
        self.num_steps = 1000
        self.x_0 = torch.randn([self.batch_size, self.data_dim])
        self.model = DummyModel()
        self.original_scheduler = CondOTScheduler()
        self.new_scheduler = CosineScheduler()

    def test_schedule_transformation(self):
        solver_original = ODESolver(velocity_model=self.model)
        x_1_original = solver_original.sample(
            time_steps=torch.tensor([0.0, 1.0]),
            x_init=self.x_0,
            step_size=1 / self.num_steps,
            method="euler",
        )[1]
        transformed_model = ScheduleTransformedModel(
            velocity_model=self.model,
            original_scheduler=self.original_scheduler,
            new_scheduler=self.new_scheduler,
        )

        solver_transformed = ODESolver(velocity_model=transformed_model)
        x_1_transformed = solver_transformed.sample(
            time_steps=torch.tensor([0.0, 1.0]),
            x_init=self.x_0,
            step_size=1 / self.num_steps,
            method="euler",
        )[1]

        self.assertTrue(
            torch.allclose(x_1_original, x_1_transformed, atol=1e-2),
            "The samples with and without the transformed scheduler should be approximately equal.",
        )


if __name__ == "__main__":
    unittest.main()
