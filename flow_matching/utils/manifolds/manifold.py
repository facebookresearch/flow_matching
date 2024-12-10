# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch.nn as nn
from torch import Tensor


class Manifold(nn.Module, metaclass=abc.ABCMeta):
    """A manifold class that contains projection operations and logarithm and exponential maps."""

    @abc.abstractmethod
    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        r"""Computes exponential map :math:`\exp_x(u)`.

        Args:
            x (Tensor): point on the manifold
            u (Tensor): tangent vector at point :math:`x`

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Tensor: transported point
        """
        raise NotImplementedError

    @abc.abstractmethod
    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Computes logarithmic map :math:`\log_x(y)`.

        Args:
            x (Tensor): point on the manifold
            y (Tensor): point on the manifold

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Tensor: tangent vector at point :math:`x`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def projx(self, x: Tensor) -> Tensor:
        """Project point :math:`x` on the manifold.

        Args:
            x (Tensor): point to be projected

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Tensor: projected point on the manifold
        """
        raise NotImplementedError

    @abc.abstractmethod
    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Project vector :math:`u` on a tangent space for :math:`x`.

        Args:
            x (Tensor): point on the manifold
            u (Tensor): vector to be projected

        Raises:
            NotImplementedError: if not implemented

        Returns:
            Tensor: projected tangent vector
        """
        raise NotImplementedError


class Euclidean(Manifold):
    """The Euclidean manifold."""

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        return x + u

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        return y - x

    def projx(self, x: Tensor) -> Tensor:
        return x

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        return u
