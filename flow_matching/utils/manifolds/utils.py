# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from torch import Tensor

from flow_matching.utils.manifolds import Manifold


def geodesic(
    manifold: Manifold, start_point: Tensor, end_point: Tensor
) -> Callable[[Tensor], Tensor]:
    """Generate parameterized function for geodesic curve.

    Args:
        manifold (Manifold): the manifold to compute geodesic on.
        start_point (Tensor): point on the manifold at :math:`t=0`.
        end_point (Tensor): point on the manifold at :math:`t=1`.

    Returns:
        Callable[[Tensor], Tensor]: a function that takes in :math:`t` and outputs the geodesic at time :math:`t`.
    """

    shooting_tangent_vec = manifold.logmap(start_point, end_point)

    def path(t: Tensor) -> Tensor:
        """Generate parameterized function for geodesic curve.

        Args:
            t (Tensor): Times at which to compute points of the geodesics.

        Returns:
            Tensor: geodesic path evaluated at time t.
        """
        tangent_vecs = torch.einsum("i,...k->...ik", t, shooting_tangent_vec)
        points_at_time_t = manifold.expmap(start_point.unsqueeze(-2), tangent_vecs)

        return points_at_time_t

    return path
