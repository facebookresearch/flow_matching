# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from typing import Optional, Tuple

import torch
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath, ProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from torch import Tensor
from torch.nn.modules.loss import _Loss


class SourceDistribution(ABC):
    def __init__(
        self,
    ) -> None:
        ...

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        ...

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        ...


class MaskedSourceDistribution(SourceDistribution):
    def __init__(self, mask_token: int) -> None:
        self.mask_token = mask_token

    @property
    def masked(self) -> bool:
        return True

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        return torch.zeros(tensor_size, device=device).fill_(self.mask_token).long()

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.zeros_like(tensor_like).fill_(self.mask_token).long()


class UniformSourceDistribution(SourceDistribution):
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    @property
    def masked(self) -> bool:
        return False

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        return torch.randint(size=tensor_size, high=self.vocab_size, device=device)

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.randint_like(tensor_like, high=self.vocab_size)


def get_path(scheduler_type: str, exponent: Optional[float] = None) -> ProbPath:
    if scheduler_type == "polynomial":
        scheduler = PolynomialConvexScheduler(n=exponent)
    else:
        raise ValueError(f"{scheduler_type} is not supported")

    return MixtureDiscreteProbPath(scheduler=scheduler)


def get_source_distribution(
    source_distribution: str, vocab_size: int
) -> SourceDistribution:
    if source_distribution == "mask":
        return MaskedSourceDistribution(mask_token=vocab_size)
    elif source_distribution == "uniform":
        return UniformSourceDistribution(vocab_size=vocab_size)
    else:
        raise ValueError(f"{source_distribution} is not supported")


def get_loss_function(loss_function: str, path: Optional[ProbPath] = None) -> _Loss:
    if loss_function == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif loss_function == "generalized_kl":
        assert path is not None

        return MixturePathGeneralizedKL(path=path)
    else:
        raise ValueError(f"{loss_function} is not supported")
