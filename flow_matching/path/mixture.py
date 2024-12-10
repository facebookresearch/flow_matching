# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from torch import Tensor

from flow_matching.path.path import ProbPath

from flow_matching.path.path_sample import DiscretePathSample
from flow_matching.path.scheduler import ConvexScheduler
from flow_matching.utils import expand_tensor_like, unsqueeze_to_match


class MixtureDiscreteProbPath(ProbPath):
    r"""The ``MixtureDiscreteProbPath`` class defines a factorized discrete probability path.

    This path remains constant at the source data point :math:`X_0` until a random time, determined by the scheduler, when it flips to the target data point :math:`X_1`.
    The scheduler determines the flip probability using the parameter :math:`\sigma_t`, which is a function of time `t`. Specifically, :math:`\sigma_t` represents the probability of remaining at :math:`X_0`, while :math:`1 - \sigma_t` is the probability of flipping to :math:`X_1`:

    .. math::

        P(X_t = X_0) = \sigma_t \quad \text{and} \quad  P(X_t = X_1) = 1 - \sigma_t,

    where :math:`\sigma_t` is provided by the scheduler.

    Example:

    .. code-block:: python

        >>> x_0 = torch.zeros((1, 3, 3))
        >>> x_1 = torch.ones((1, 3, 3))

        >>> path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))
        >>> result = path.sample(x_0, x_1, t=torch.tensor([0.1])).x_t
        >>> result
        tensor([[[0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0]]])

        >>> result = path.sample(x_0, x_1, t=torch.tensor([0.5])).x_t
        >>> result
        tensor([[[1.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0]]])

        >>> result = path.sample(x_0, x_1, t=torch.tensor([1.0])).x_t
        >>> result
        tensor([[[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]]])

    Args:
        scheduler (ConvexScheduler): The scheduler that provides :math:`\sigma_t`.
    """

    def __init__(self, scheduler: ConvexScheduler):
        assert isinstance(
            scheduler, ConvexScheduler
        ), "Scheduler for ConvexProbPath must be a ConvexScheduler."

        self.scheduler = scheduler

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> DiscretePathSample:
        r"""Sample from the affine probability path:
            | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)` and a scheduler :math:`(\alpha_t,\sigma_t)`.
            | return :math:`X_0, X_1, t`, and :math:`X_t \sim p_t`.
        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            DiscretePathSample: a conditional sample at :math:`X_t ~ p_t`.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        sigma_t = self.scheduler(t).sigma_t

        sigma_t = expand_tensor_like(input_tensor=sigma_t, expand_to=x_1)

        source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t
        x_t = torch.where(condition=source_indices, input=x_0, other=x_1)

        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)

    def posterior_to_velocity(
        self, posterior_logits: Tensor, x_t: Tensor, t: Tensor
    ) -> Tensor:
        r"""Convert the factorized posterior to velocity.

        | given :math:`p(X_1|X_t)`. In the factorized case: :math:`\prod_i p(X_1^i | X_t)`.
        | return :math:`u_t`.

        Args:
            posterior_logits (Tensor): logits of the x_1 posterior conditional on x_t, shape (..., vocab size).
            x_t (Tensor): path sample at time t, shape (...).
            t (Tensor): time in [0,1].

        Returns:
            Tensor: velocity.
        """
        posterior = torch.softmax(posterior_logits, dim=-1)
        vocabulary_size = posterior.shape[-1]
        x_t = F.one_hot(x_t, num_classes=vocabulary_size)
        t = unsqueeze_to_match(source=t, target=x_t)

        scheduler_output = self.scheduler(t)

        kappa_t = scheduler_output.alpha_t
        d_kappa_t = scheduler_output.d_alpha_t

        return (d_kappa_t / (1 - kappa_t)) * (posterior - x_t)
