# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

import math
from collections import Counter
from typing import List

import torch
import torch.nn.functional as F
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath, ProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.utils import ModelWrapper
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from logic.flow import SourceDistribution


class WrappedModel(ModelWrapper):
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        return self.model(x_t=x, time=t).float()


@torch.no_grad()
def compute_perplexity(samples: Tensor, perplexity_batch_size: int) -> Tensor:
    eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(samples.device).eval()
    batches = samples.shape[0] // perplexity_batch_size
    total_perplexity = 0

    for i in range(batches):
        s = samples[i * perplexity_batch_size : (i + 1) * perplexity_batch_size]
        _, logits = eval_model(s, labels=s)[:2]
        logits = logits.transpose(-1, -2).detach()

        perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none")
        perplexity = perplexity.mean(dim=-1).exp().mean()

        total_perplexity += perplexity

    total_perplexity /= batches

    return total_perplexity


def _sample_entropy(sample: List) -> float:
    histogram = Counter(sample)
    total = sum(histogram.values())
    entropy = 0

    for count in histogram.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def compute_entropy(samples: Tensor) -> Tensor:
    entropies = [_sample_entropy(sample.tolist()) for sample in samples]
    entropy = sum(entropies) / len(entropies)

    return torch.tensor(entropy, device=samples.device)


@torch.no_grad()
def estimate_likelihood(
    model: nn.Module,
    dataloader: DataLoader,
    source_distribution: SourceDistribution,
    path: ProbPath,
    n_discretization: int,
    device: torch.device,
    batch_size: int = 32,
    epsilon: float = 1e-3,
) -> Tensor:
    model = WrappedModel(model)

    # Generalized KL function (will use it to compute the elbo)
    linear_scheduler = PolynomialConvexScheduler(n=1.0)
    linear_path = MixtureDiscreteProbPath(scheduler=linear_scheduler)

    generalized_kl_fn = MixturePathGeneralizedKL(path=linear_path, reduction="none")

    # Time discretization
    discretization = (
        torch.linspace(0, 1, n_discretization + 1, device=device)[:-1]
        .view(-1, 1)
        .repeat(1, batch_size)
    )

    elbo = torch.zeros((1,), device=device)
    n_elements = torch.zeros((1,), device=device)

    for x_1 in tqdm(dataloader, total=len(dataloader)):
        x_1 = x_1["input_ids"].to(device)

        # Lower variance estimator for time discretization
        discretization = discretization + torch.rand(
            size=(1, batch_size), device=device
        )
        discretization = discretization % 1
        discretization = discretization * (1 - epsilon)

        for k in discretization[:, : x_1.shape[0]]:
            x_0 = source_distribution.sample_like(x_1)
            x_t = linear_path.sample(t=k, x_0=x_0, x_1=x_1).x_t

            t = path.scheduler.kappa_inverse(k)

            logits = model(x=x_t, t=t)

            generalized_kl = generalized_kl_fn(logits=logits, x_1=x_1, x_t=x_t, t=k)
            n_elements += generalized_kl.numel()

            elbo += generalized_kl.sum()

    return elbo, n_elements
