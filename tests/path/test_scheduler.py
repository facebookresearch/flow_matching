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
    LinearVPScheduler,
    PolynomialConvexScheduler,
    SchedulerOutput,
    VPScheduler,
)
from torch import Tensor


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.t = torch.tensor([0.1, 0.5, 0.9])

    def assert_output_shapes(
        self, outputs: SchedulerOutput, expected_shape: torch.Size
    ):
        self.assertEqual(outputs.alpha_t.shape, expected_shape)
        self.assertEqual(outputs.sigma_t.shape, expected_shape)
        self.assertEqual(outputs.d_alpha_t.shape, expected_shape)
        self.assertEqual(outputs.d_sigma_t.shape, expected_shape)

    def assert_recover_t_from_kappa(self, scheduler, t: Tensor):
        scheduler_output = scheduler(t)
        t_recovered = scheduler.kappa_inverse(scheduler_output.alpha_t)

        self.assertTrue(
            torch.allclose(t, t_recovered, atol=1e-5),
            f"Recovered t: {t_recovered}, Original t: {t}",
        )

    def assert_recover_t_from_snr(self, scheduler, t: Tensor):
        scheduler_output = scheduler(t)
        snr = scheduler_output.alpha_t / scheduler_output.sigma_t

        t_recovered = scheduler.snr_inverse(snr)

        self.assertTrue(
            torch.allclose(t, t_recovered, atol=1e-5),
            f"Recovered t: {t_recovered}, Original t: {t}",
        )

    def test_cond_ot_scheduler(self):
        scheduler = CondOTScheduler()
        outputs = scheduler(self.t)

        self.assert_output_shapes(outputs, self.t.shape)

        self.assert_recover_t_from_kappa(scheduler, self.t)
        self.assert_recover_t_from_snr(scheduler, self.t)

    def test_cosine_scheduler(self):
        scheduler = CosineScheduler()
        outputs = scheduler(self.t)
        self.assert_output_shapes(outputs, self.t.shape)

        self.assert_recover_t_from_snr(scheduler, self.t)

    def test_scheduler_vp(self):
        scheduler = VPScheduler()
        outputs = scheduler(self.t)
        self.assert_output_shapes(outputs, self.t.shape)

        self.assert_recover_t_from_snr(scheduler, self.t)

    def test_scheduler_vp_linear(self):
        scheduler = LinearVPScheduler()
        outputs = scheduler(self.t)
        self.assert_output_shapes(outputs, self.t.shape)

        self.assert_recover_t_from_snr(scheduler, self.t)

    def test_polynomial_convex_scheduler(self):
        scheduler = PolynomialConvexScheduler(n=2)
        outputs = scheduler(self.t)
        self.assert_output_shapes(outputs, self.t.shape)

        self.assert_recover_t_from_kappa(scheduler, self.t)
        self.assert_recover_t_from_snr(scheduler, self.t)


if __name__ == "__main__":
    unittest.main()
