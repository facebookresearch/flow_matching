# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn, Tensor

# flow_matching
from flow_matching.loss.generalized_loss import MixturePathGeneralizedKL
from flow_matching.path.mixture import MixtureDiscreteProbPath
from flow_matching.path.scheduler import Scheduler
from flow_matching.path.scheduler.schedule_transform import ScheduleTransformedModel
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper


def _default_continuous_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Mean squared error loss for continuous modalities.

    Args:
        pred (Tensor): predicted velocity field.
        target (Tensor): target velocity field.

    Returns:
        Tensor: mean squared error loss.
    """
    return torch.mean((pred - target) ** 2)


class Flow(nn.Module):
    """
    Generic multimodal flow matching model.

    This class aggregates multiple modalities, each with its own model, path,
    scheduler, and loss. It provides utilities for training (computing the total
    loss) and inference (sampling) across all modalities.

    Args:
        modalities (Dict[str, Dict[str, Any]]):
            Mapping from modality name to a dict with keys:
                - "model": nn.Module (or ModelWrapper) that implements the velocity model.
                - "path": a probability path object (e.g., MixtureDiscreteProbPath for discrete data,
                or any continuous path implementation).
                - "loss" (optional): a callable loss function. If omitted, a default loss is chosen
                based on the path type.
        training_scheduler (Optional[Scheduler]): Scheduler used during training.
        inference_scheduler (Optional[Scheduler]): Scheduler used during inference (sampling).

    Raises:
        TypeError: if any model is not an instance of nn.Module.
    """

    def __init__(
        self,
        modalities: Dict[str, Dict[str, Any]],
        training_scheduler: Optional[Scheduler] = None,
        inference_scheduler: Optional[Scheduler] = None,
    ) -> None:
        super().__init__()
        self.modalities = nn.ModuleDict()
        self.paths: Dict[str, Any] = {}
        self.loss_fns: Dict[str, Callable] = {}
        self.training_scheduler = training_scheduler
        self.inference_scheduler = inference_scheduler

        for name, spec in modalities.items():
            model = spec["model"]
            if not isinstance(model, nn.Module):
                raise TypeError(f"Model for modality '{name}' must be an nn.Module.")
            self.modalities[name] = model

            path = spec["path"]
            self.paths[name] = path

            # Choose loss function
            loss_fn = spec.get("loss")
            if loss_fn is None:
                if isinstance(path, MixtureDiscreteProbPath):
                    loss_fn = MixturePathGeneralizedKL(path)
                else:
                    loss_fn = _default_continuous_loss
            self.loss_fns[name] = loss_fn

    def training_loss(
        self,
        inputs: Dict[str, Tuple[Tensor, Tensor]],
        t: Tensor,
    ) -> Tensor:
        """
        Compute the total training loss across all modalities.

        Args:
            inputs (Dict[str, Tuple[Tensor, Tensor]]): Mapping from modality name to a tuple ``(x_1, x_t)`` where ``x_1`` is the data at
                time ``0`` and ``x_t`` is the data at the sampled time ``t``.
            t (Tensor): Tensor of shape ``(batch,)`` containing the time values.

        Returns:
            Tensor: scalar loss (sum of modality losses).
        """
        total_loss = 0.0
        for name, (x_1, x_t, dx_t) in inputs.items():
            model = self.modalities[name]
            path = self.paths[name]
            loss_fn = self.loss_fns[name]

            if isinstance(path, MixtureDiscreteProbPath):
                # Discrete case: model should output logits.
                logits = model(x=x_t, t=t)
                loss = loss_fn(logits, x_1, x_t, t)
            else:
                # Continuous case: model returns velocity field.
                pred_vel = model(x=x_t, t=t)
                loss = loss_fn(pred_vel, dx_t)

            total_loss = total_loss + loss

        return total_loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        steps: int = 1000,
    ) -> Dict[str, Tensor]:
        """
        Generate samples for each modality using the inference scheduler.

        Args:
            batch_size (int): Number of samples to generate.
            device (torch.device, optional): Device on which to run the sampling.
            steps (int, optional): Number of integration steps for the ODE solver.

        Returns:
            Dict[str, Tensor]: mapping from modality name to sampled tensor.
        """
        xs: Dict[str, Tensor] = {}
        for name, model in self.modalities.items():
            path = self.paths[name]

            # Maybe transform the schedule of each modality.
            velocity_model = model
            if (
                self.training_scheduler is not None
                and self.inference_scheduler is not None
            ):
                velocity_model = ScheduleTransformedModel(
                    velocity_model=model,
                    original_scheduler=self.training_scheduler,
                    new_scheduler=self.inference_scheduler,
                )

            # Initialise samples for each modality.
            assert hasattr(
                model, "sample_shape"
            ), f"Model for modality '{name}' must implement 'sample_shape' method."
            assert hasattr(
                model, "sample_prior"
            ), f"Model for modality '{name}' must implement 'sample_prior' method."
            x_shape = model.sample_shape(batch_size)
            xs[name] = model.sample_prior(x_shape, device=device)

            # Set up ODE solver.
            solver = ODESolver(velocity_model=velocity_model)
            if isinstance(path, MixtureDiscreteProbPath):

                class WrappedModel(ModelWrapper):
                    """Wrap velocity model to output probabilities."""

                    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
                        """Output class probabilities."""
                        return torch.softmax(self.model(x, t, **extras), dim=-1)

                wrapped_probability_denoiser = WrappedModel(velocity_model)
                solver = MixtureDiscreteEulerSolver(
                    model=wrapped_probability_denoiser,
                    path=path,
                    vocabulary_size=wrapped_probability_denoiser.model.input_dim,
                )

            # Solve ODE to obtain samples at time 1.
            time_grid = torch.linspace(0.0, 1.0, steps, device=device)
            xs[name] = solver.sample(
                x_init=xs[name],
                step_size=1.0 / steps,
                time_grid=time_grid,
            )

        return xs
