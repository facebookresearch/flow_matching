# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor

from torch.nn import functional as F

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.solver.solver import Solver
from flow_matching.solver.utils import get_nearest_times
from flow_matching.utils import categorical, ModelWrapper

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class MultimodalSolver(Solver):
    """Solver for multiple continuous and discrete data modalities.

    This solver handles an arbitrary number of modalities, which can be either
    continuous or discrete. Each modality has its own state tensor.
    All modalities share the same time discretization and are updated
    simultaneously at each step.

    For continuous modalities, an Euler integration step is used. For discrete
    modalities, the update follows the procedure from `MixtureDiscreteEulerSolver`.

    Args:
        model (Union[ModelWrapper, Callable]):
            A model that receives a sequence of state tensors
            (one per modality) as ``x`` and a scalar time tensor ``t``,
            and returns a sequence of output tensors. For continuous modalities,
            the output is a velocity. For discrete modalities, it is the
            posterior probability `p_1t`.
        modality_configs (List[Dict[str, Any]]):
            A list of configuration dictionaries, one for each modality.
            Each dictionary must have a ``'type'`` key, which is either
            ``'continuous'`` or ``'discrete'``. Discrete modality configs must
            also provide a ``'path'`` key with a `MixtureDiscreteProbPath` object.
        source_distribution_p (Optional[Tensor], optional): Source distribution,
            must be of shape [vocabulary_size]. Required only when divergence-free
            term for the probability velocity is non-zero. Defaults to None.

    Raises:
        TypeError: If `model` is not callable.
    """

    def __init__(
        self,
        model: Union[ModelWrapper, Callable],
        modality_configs: List[Dict[str, Any]],
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        if not callable(model):
            raise TypeError(f"model must be callable, got {type(model)}")
        self.model = model
        self.modality_configs = modality_configs
        self.source_distribution_p = source_distribution_p

        self._validate_configs()

    def _validate_configs(self):
        """Validates the modality configurations."""
        if not isinstance(self.modality_configs, list):
            raise TypeError("modality_configs must be a list of dictionaries.")
        for i, config in enumerate(self.modality_configs):
            if not isinstance(config, dict):
                raise TypeError(f"Config for modality {i} must be a dictionary.")
            if "type" not in config:
                raise ValueError(f"Config for modality {i} must have a 'type' key.")
            if config["type"] not in ["continuous", "discrete"]:
                raise ValueError(
                    f"Unsupported modality type '{config['type']}' for modality {i}."
                )
            if config["type"] == "discrete":
                if "path" not in config:
                    raise ValueError(
                        f"Discrete modality {i} requires a 'path' in its config."
                    )
                if not isinstance(config["path"], MixtureDiscreteProbPath):
                    raise TypeError(
                        f"'path' for discrete modality {i} must be a MixtureDiscreteProbPath instance."
                    )

    def sample(
        self,
        x_init: Sequence[Tensor],
        step_size: Optional[float],
        div_free: Union[float, Callable[[float], float]] = 0.0,
        method: str = "euler",
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
        verbose: bool = False,
        **model_extras: dict,
    ) -> Union[Sequence[Tensor], Sequence[List[Tensor]]]:
        """Sample all modalities simultaneously.

        Args:
            x_init (Sequence[Tensor]): Initial states for each modality.
            step_size (Optional[float]): Fixed step size for uniform discretization.
                If ``None``, the discretization is taken from ``time_grid``.
            div_free (Union[float, Callable[[float], float]]): The coefficient
                of the divergence-free term in the probability velocity
                (for discrete modalities). Can be either a float or a time
                dependent function. Defaults to 0.0.
            method (str): Numerical integration method. Currently only ``"euler"`` is
                supported, representing a single forward step.
            time_grid (Tensor): Tensor of time points defining the interval.
            return_intermediates (bool): If ``True``, returns a list of tensors for
                each modality containing the state at each intermediate time step.
            enable_grad (bool): Whether to enable gradient tracking during integration.
            verbose (bool): If ``True``, displays a progress bar during sampling.
            **model_extras (dict): Additional arguments passed to the model.

        Raises:
            ValueError: If the number of initial states does not match the number of
                modality configurations.
            NotImplementedError: If an unsupported integration method is specified.
            ImportError: If ``verbose`` is ``True`` but ``tqdm`` is not installed.
            TypeError: If the model's output does not match the expected format.

        Returns:
            Union[Sequence[Tensor], Sequence[List[Tensor]]]: If ``return_intermediates`` is
            ``False`` (default), returns a list of final state tensors, one per
            modality. If ``True``, returns a list where each element is another
            list of tensors representing the trajectory for a modality.
        """
        if len(x_init) != len(self.modality_configs):
            raise ValueError(
                "Number of initial states must match the number of modality configurations."
            )
        if method != "euler":
            raise NotImplementedError(
                f"Method '{method}' is not implemented for MultimodalSolver."
            )
        if not div_free == 0.0:
            assert (
                self.source_distribution_p is not None
            ), "Source distribution p must be specified in order to add a divergence-free term to the probability velocity for each discrete modality."

        # Initialize the current state `x_t` with the initial state `X_0`.
        device = x_init[0].device
        batch_size = x_init[0].shape[0]
        time_grid = time_grid.to(device)

        if step_size is None:
            # If step_size is None then set the t discretization to time_grid.
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            # If step_size is float then t discretization is uniform with step size set by step_size.
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (
                t_final - t_init
            ) > step_size, f"Time interval [time_grid[0], time_grid[-1]] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=device,
            )

            if return_intermediates:
                # Get order of intermediate steps
                order = torch.argsort(time_grid)
                # Compute intermediate steps to return via nearest points in t_discretization to time_grid.
                time_grid = get_nearest_times(
                    time_grid=time_grid, t_discretization=t_discretization
                )

        states: Sequence[Tensor] = [(x if enable_grad else x.clone()) for x in x_init]
        intermediates: Sequence[List[Tensor]] = (
            [[x if enable_grad else x.clone()] for x in x_init]
            if return_intermediates
            else []
        )

        steps_counter = 0

        if verbose:
            if not TQDM_AVAILABLE:
                raise ImportError(
                    "tqdm is required for verbose mode. Please install it."
                )
            ctx = tqdm(total=t_final, desc=f"NFE: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx, torch.set_grad_enabled(enable_grad):
            for i in range(n_steps):
                # NOTE: For now, all modalities share the same time
                t = [t_discretization[i : i + 1].repeat(batch_size)] * len(states)
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                outputs = self.model(states, t, **model_extras)

                if not isinstance(outputs, (list, tuple)) or len(outputs) != len(
                    states
                ):
                    raise TypeError(
                        "The model must return a sequence of tensors matching the number of modalities."
                    )

                for idx, config in enumerate(self.modality_configs):
                    model_output = outputs[idx]

                    if config["type"] == "continuous":
                        # Sample x_{t+h} = x_t + h * v(x_t,t)
                        states[idx] = states[idx] + h * model_output

                    elif config["type"] == "discrete":
                        dtype = config.get("dtype_categorical", torch.float32)

                        # Sample x_1 ~ p_1|t( \cdot |x_t)
                        p_1t = torch.softmax(model_output, dim=-1)
                        x_1 = categorical(p_1t.to(dtype=dtype))

                        # Checks if final step
                        if i == n_steps - 1:
                            states[idx] = x_1  # x_t = x_1 at final step
                        else:
                            vocabulary_size = p_1t.shape[-1]
                            if self.source_distribution_p is not None:
                                assert self.source_distribution_p.shape == torch.Size(
                                    [vocabulary_size]
                                ), f"Source distribution p dimension must match the vocabulary size {vocabulary_size}. Got {self.source_distribution_p.shape}."

                            # Compute u_t(x|x_t,x_1)
                            path: MixtureDiscreteProbPath = config["path"]

                            t_expanded = t[idx].reshape(
                                -1, *[1] * (model_output.dim() - 1)
                            )
                            scheduler_output = path.scheduler(t=t_expanded)

                            k_t = scheduler_output.alpha_t
                            d_k_t = scheduler_output.d_alpha_t

                            delta_1 = F.one_hot(x_1, num_classes=vocabulary_size).to(
                                k_t.dtype
                            )
                            u = d_k_t / (1 - k_t) * delta_1

                            # Add divergence-free part
                            div_free_t = (
                                div_free(t[idx]) if callable(div_free) else div_free
                            )

                            if div_free_t > 0:
                                p_0 = self.source_distribution_p[
                                    (None,) * states[idx].dim()
                                ]
                                u = u + div_free_t * d_k_t / (k_t * (1 - k_t)) * (
                                    (1 - k_t) * p_0 + k_t * delta_1
                                )

                            # Set u_t(x_t|x_t,x_1) = 0
                            delta_t = F.one_hot(
                                states[idx], num_classes=vocabulary_size
                            )
                            u = torch.where(
                                delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                            )

                            # Sample x_t ~ u_t( \cdot |x_t,x_1)
                            intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                            mask_jump = torch.rand(
                                size=states[idx].shape, device=states[idx].device
                            ) < 1 - torch.exp(-h * intensity)

                            if mask_jump.sum() > 0:
                                states[idx][mask_jump] = categorical(
                                    u[mask_jump].to(dtype=dtype)
                                )

                    # Increment time for each modality
                    t[idx] = t[idx] + h

                steps_counter += 1

                if return_intermediates:
                    for idx, s in enumerate(states):
                        if t[idx] in time_grid:
                            intermediates[idx].append(s if enable_grad else s.clone())

                if verbose:
                    ctx.n = (torch.cat(t) * n_steps).mean().long().item()
                    ctx.refresh()
                    ctx.set_description(f"NFE: {steps_counter}")

        if return_intermediates:
            if step_size is None:
                return intermediates
            else:
                return [
                    [intermediates[idx][i] for i in order]
                    for idx in range(len(intermediates))
                ]
        else:
            return states
