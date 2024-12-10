# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class PathSample:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        x_1 (Tensor): the target sample :math:`X_1`.
        x_0 (Tensor): the source sample :math:`X_0`.
        t (Tensor): the time sample :math:`t`.
        x_t (Tensor): samples :math:`X_t \sim p_t(X_t)`, shape (Batch, ...).
        dx_t (Tensor): conditional target :math:`\frac{\partial X}{\partial t}`, shape: (Batch, ...).

    """

    x_1: Tensor = field(metadata={"help": "target samples X_1 (Batch, ...)."})
    x_0: Tensor = field(metadata={"help": "source samples X_0 (Batch, ...)."})
    t: Tensor = field(metadata={"help": "time samples t (Batch, ...)."})
    x_t: Tensor = field(
        metadata={"help": "samples x_t ~ p_t(X_t), shape (Batch, ...)."}
    )
    dx_t: Tensor = field(
        metadata={"help": "conditional target dX_t, shape: (Batch, ...)."}
    )


@dataclass
class DiscretePathSample:
    """
    Represents a sample of a conditional-flow generated discrete probability path.

    Attributes:
        x_1 (Tensor): the target sample :math:`X_1`.
        x_0 (Tensor): the source sample :math:`X_0`.
        t (Tensor): the time sample  :math:`t`.
        x_t (Tensor): the sample along the path  :math:`X_t \sim p_t`.
    """

    x_1: Tensor = field(metadata={"help": "target samples X_1 (Batch, ...)."})
    x_0: Tensor = field(metadata={"help": "source samples X_0 (Batch, ...)."})
    t: Tensor = field(metadata={"help": "time samples t (Batch, ...)."})
    x_t: Tensor = field(
        metadata={"help": "samples X_t ~ p_t(X_t), shape (Batch, ...)."}
    )
