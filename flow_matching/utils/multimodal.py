# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor

# flow_matching
from flow_matching.loss.generalized_loss import MixturePathGeneralizedKL
from flow_matching.path.mixture import MixtureDiscreteProbPath
from flow_matching.solver.multimodal_solver import MultimodalSolver


MULTIMODAL_METHOD = Literal["euler"]


def _default_continuous_loss(
    pred: Tensor, target: Tensor, reduction: str = "none"
) -> Tensor:
    """
    Mean squared error loss for continuous modalities.

    Args:
        pred (Tensor): predicted velocity field.
        target (Tensor): target velocity field.
        reduction (str): reduction method, one of 'mean', 'sum', or 'none'.

    Raises:
        ValueError: if reduction is not one of 'none', 'mean', or 'sum'.

    Returns:
        Tensor: computed loss.
    """
    loss = (pred - target) ** 2

    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")


class Flow(nn.Module):
    """
    Generic multimodal flow matching model.

    This class aggregates multiple modalities, each with its own model, path,
    scheduler, and loss. It provides utilities for training (computing the total
    loss) and inference (sampling) across all modalities.

    Args:
        model (nn.Module):
            A model that receives a sequence of state tensors
            (one per modality) as ``x`` and a scalar time tensor ``t``,
            and returns a sequence of output tensors. For continuous modalities,
            the output is a velocity. For discrete modalities, it is the
            posterior probability `p_1t`.
        modalities (Dict[str, Dict[str, Any]]):
            Mapping from modality name to a dict with keys:
                - "path": A probability path object (e.g., MixtureDiscreteProbPath for discrete data,
                or any continuous path implementation).
                - "loss" (optional): A callable loss function. If omitted, a default loss is chosen
                based on the path type.
                - "weight" (optional): A float weight for the modality's training loss. Defaults to 1.0.
    """

    def __init__(
        self,
        model: nn.Module,
        modalities: Dict[str, Dict[str, Any]],
    ) -> None:
        super().__init__()
        self.model = model
        self.paths: Dict[str, Any] = {}
        self.loss_fns: Dict[str, Callable] = {}
        self.loss_weights: Dict[str, float] = {}

        for name, spec in modalities.items():
            path = spec["path"]
            self.paths[name] = path

            # Choose loss function
            loss_fn = spec.get("loss")
            if loss_fn is None:
                if isinstance(path, MixtureDiscreteProbPath):
                    loss_fn = MixturePathGeneralizedKL(path, reduction="none")
                else:
                    loss_fn = _default_continuous_loss
            self.loss_fns[name] = loss_fn
            self.loss_weights[name] = spec.get("weight", 1.0)

    def training_loss(
        self,
        x_1: Sequence[Tensor],
        x_t: Sequence[Tensor],
        dx_t: Sequence[Tensor],
        t: Sequence[Tensor],
        detach_loss_dict: bool = True,
        **model_extras: dict,
    ) -> Tuple[Sequence[Tensor], Dict[str, Tensor]]:
        """
        Compute the total training loss across all modalities.

        Args:
            x_1 (Sequence[Tensor]): Sequence of tensors, one per modality,
                containing the data at time 1.
            x_t (Sequence[Tensor]): Sequence of tensors, one per modality,
                containing the data at time t.
            dx_t (Sequence[Tensor]): Sequence of tensors, one per modality,
                containing the velocity field at time t.
            t (Sequence[Tensor]): Sequence of tensors, one per modality,
                containing the time values.
            detach_loss_dict (bool): If ``True``, detaches individual modality losses
                from the computation graph when storing them in the loss dictionary.
                Defaults to ``True``.
            **model_extras (dict): Additional keyword arguments to pass to the model.

        Returns:
            Tuple[Sequence[Tensor], Dict[str, Tensor]]:
                Scalar loss (sum of modality losses) and a dictionary
                of individual modality losses.
        """
        assert (
            len(x_1) == len(x_t) == len(dx_t) == len(t) == len(self.paths)
        ), "Input sequences must match the number of modalities."

        loss_dict = {}
        total_loss = 0.0

        logits = self.model(x_t, t, **model_extras)

        for i, name in enumerate(self.paths):
            path = self.paths[name]
            loss_fn = self.loss_fns[name]

            if isinstance(path, MixtureDiscreteProbPath):
                # Discrete case: model should output logits.
                assert x_t[i].dtype == torch.long, (
                    f"Expected integer tensor for discrete modality '{name}', "
                    f"got {x_t[i].dtype}",
                )
                loss = loss_fn(logits[i], x_1[i], x_t[i], t[i])
            else:
                # Continuous case: model returns velocity field.
                assert x_t[i].dtype == torch.float32, (
                    f"Expected float tensor for continuous modality '{name}', "
                    f"got {x_t[i].dtype}",
                )
                loss = loss_fn(logits[i], dx_t[i])

            loss_dict[name] = loss.detach() if detach_loss_dict else loss
            weight = self.loss_weights.get(name, 1.0)
            total_loss = total_loss + loss.mean() * weight

        return total_loss, loss_dict

    def sample(
        self,
        x_init: Sequence[Tensor],
        time_grid: Optional[Tensor] = None,
        device: torch.device = torch.device("cpu"),
        steps: int = 1000,
        step_size: Optional[float] = None,
        div_free: Union[float, Callable[[float], float]] = 0.0,
        method: MULTIMODAL_METHOD = "euler",
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras: dict,
    ) -> Union[Sequence[Tensor], Sequence[List[Tensor]]]:
        """
        Generate samples for each modality using the inference scheduler.

        Args:
            x_init (Sequence[Tensor]):
                Sequence of tensors, one per modality, containing the initial states at time 0.
                For continuous modalities, this is typically Gaussian noise.
                For discrete modalities, this is typically samples from a uniform categorical distribution.
            time_grid (Optional[Tensor]): Optional tensor of time points defining the interval.
                If provided, it overrides the uniform discretization defined by `steps`.
            device (torch.device, optional): Device on which to run the sampling.
            steps (int, optional): Number of integration steps for the ODE solver.
            step_size (Optional[float]): Fixed step size for uniform discretization.
                If ``None``, the step size is computed from ``steps``.
            div_free (Union[float, Callable[[float], float]]): The coefficient
                of the divergence-free term in the probability velocity
                (for discrete modalities). Can be either a float or a time
                dependent function. Defaults to 0.0.
            method (MULTIMODAL_METHOD): Numerical integration method. Currently only ``"euler"`` is
                supported, representing a single forward step.
            return_intermediates (bool): If ``True``, returns a list of tensors for
                each modality containing the state at each intermediate time step.
            enable_grad (bool): Whether to enable gradient tracking during integration.
            **model_extras (dict): Additional keyword arguments to pass to the model.

        Returns:
            Union[Sequence[Tensor], Sequence[List[Tensor]]]: A list where each element corresponds to a modality.
            Each element is either a tensor of shape ``(batch_size, ...)`` containing the samples,
            or a list of tensors (if `return_intermediates` is True in `MultimodalSolver.sample`).
        """
        # Validate samples for each modality.
        for i, name in enumerate(self.paths):
            path = self.paths[name]

            if isinstance(path, MixtureDiscreteProbPath):
                assert x_init[i].dtype == torch.long, (
                    f"Expected integer tensor for discrete modality '{name}', "
                    f"got {x_init[i].dtype}",
                )
            else:
                assert x_init[i].dtype == torch.float32, (
                    f"Expected float tensor for continuous modality '{name}', "
                    f"got {x_init[i].dtype}",
                )

            x_init[i] = x_init[i].to(device)

        # Set up Euler solver for each modality.
        modality_configs = [
            {
                "name": name,
                "type": (
                    "discrete"
                    if isinstance(path, MixtureDiscreteProbPath)
                    else "continuous"
                ),
                "path": path,
            }
            for name, path in self.paths.items()
        ]
        solver = MultimodalSolver(
            model=self.model,
            modality_configs=modality_configs,
        )

        # Solve to obtain multimodal samples at time 1.
        step_size = step_size or (1.0 / steps)
        time_grid = time_grid or torch.linspace(0.0, 1.0, steps, device=device)

        samples = solver.sample(
            x_init=x_init,
            step_size=step_size,
            div_free=div_free,
            method=method,
            time_grid=time_grid,
            return_intermediates=return_intermediates,
            enable_grad=enable_grad,
            **model_extras,
        )

        return samples
