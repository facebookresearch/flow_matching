# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch import Tensor
from torch.func import jvp, vmap

from flow_matching.path.path import ProbPath

from flow_matching.path.path_sample import PathSample
from flow_matching.path.scheduler import ConvexScheduler
from flow_matching.utils import expand_tensor_like

from flow_matching.utils.manifolds import geodesic, Manifold


class GeodesicProbPath(ProbPath):
    r"""The ``GeodesicProbPath`` class represents a specific type of probability path where the transformation between distributions is defined through the geodesic path.
    Mathematically, a geodesic path can be represented as:

    .. math::

        X_t = \psi_t(X_0 | X_1) = \exp_{X_1}(\kappa_t \log_{X_1}(X_0)),

    where :math:`X_t` is the transformed data point at time `t`, :math:`X_0` and :math:`X_1` are the source and target data points, respectively, and :math:`\kappa_t` is a scheduler.

    The scheduler is responsible for providing the time-dependent :math:`\kappa_t` and must be differentiable.

    Using ``GeodesicProbPath`` in the flow matching framework:

    .. code-block:: python
        # Instantiates a manifold
        manifold = FlatTorus()

        # Instantiates a scheduler
        scheduler = CondOTScheduler()

        # Instantiates a probability path
        my_path = GeodesicProbPath(scheduler, manifold)
        mse_loss = torch.nn.MSELoss()

        for x_1 in dataset:
            # Sets x_0 to random noise
            x_0 = torch.randn()

            # Sets t to a random value in [0,1]
            t = torch.rand()

            # Samples the conditional path :math:`X_t \sim p_t(X_t|X_0,X_1)`
            path_sample = my_path.sample(x_0=x_0, x_1=x_1, t=t)

            # Computes the MSE loss w.r.t. the velocity
            loss = mse_loss(path_sample.dx_t, my_model(x_t, t))
            loss.backward()

    Args:
        scheduler (ConvexScheduler): The scheduler that provides :math:`\kappa_t`.
        manifold (Manifold): The manifold on which the probability path is defined.

    """

    def __init__(self, scheduler: ConvexScheduler, manifold: Manifold):
        self.scheduler = scheduler
        self.manifold = manifold

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        r"""Sample from the Riemannian probability path with geodesic interpolation:

        | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)` and a scheduler :math:`\kappa_t`.
        | return :math:`X_0, X_1, X_t = \exp_{X_1}(\kappa_t \log_{X_1}(X_0))`, and the conditional velocity at :math:`X_t, \dot{X}_t`.

        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            PathSample: A conditional sample at :math:`X_t \sim p_t`.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)
        t = expand_tensor_like(input_tensor=t, expand_to=x_1[..., 0:1]).clone()

        def cond_u(x_0, x_1, t):
            path = geodesic(self.manifold, x_0, x_1)
            x_t, dx_t = jvp(
                lambda t: path(self.scheduler(t).alpha_t),
                (t,),
                (torch.ones_like(t).to(t),),
            )
            return x_t, dx_t

        x_t, dx_t = vmap(cond_u)(x_0, x_1, t)
        x_t = x_t.reshape_as(x_1)
        dx_t = dx_t.reshape_as(x_1)

        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)
