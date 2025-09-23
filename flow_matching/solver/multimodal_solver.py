# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical, ModelWrapper


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

    Raises:
        TypeError: If `model` is not callable.
    """

    def __init__(
        self,
        model: Union[ModelWrapper, Callable],
        modality_configs: List[Dict[str, Any]],
    ):
        super().__init__()
        if not callable(model):
            raise TypeError(f"model must be callable, got {type(model)}")
        self.model = model
        self.modality_configs = modality_configs
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
        method: str = "euler",
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras: dict,
    ) -> Union[Sequence[Tensor], Sequence[List[Tensor]]]:
        """Sample all modalities simultaneously.

        Args:
            x_init (Sequence[Tensor]): Initial states for each modality.
            step_size (Optional[float]): Fixed step size for uniform discretization.
                If ``None``, the discretization is taken from ``time_grid``.
            method (str): Numerical integration method. Currently only ``"euler"`` is
                supported, representing a single forward step.
            time_grid (Tensor): Tensor of time points defining the interval.
            return_intermediates (bool): If ``True``, returns a list of tensors for
                each modality containing the state at each intermediate time step.
            enable_grad (bool): Whether to enable gradient tracking during integration.
            **model_extras (dict): Additional arguments passed to the model.

        Raises:
            NotImplementedError: If an unsupported integration method is specified.
            ValueError: If the number of initial states does not match the number of
                modality configurations.
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

        device = x_init[0].device
        time_grid = time_grid.to(device)

        if step_size is None:
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            t_init, t_final = time_grid[0].item(), time_grid[-1].item()
            n_steps = int(
                torch.ceil(torch.tensor((t_final - t_init) / step_size)).item()
            )
            t_discretization = torch.linspace(
                t_init, t_final, n_steps + 1, device=device
            )

        states: List[Tensor] = [x.clone() for x in x_init]
        intermediates: List[List[Tensor]] = (
            [[x.clone()] for x in x_init] if return_intermediates else []
        )

        if method != "euler":
            raise NotImplementedError(
                f"Method '{method}' is not implemented for MultimodalSolver."
            )

        with torch.set_grad_enabled(enable_grad):
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1] - t_discretization[i]

                outputs = self.model(x=states, t=t, **model_extras)

                if not isinstance(outputs, (list, tuple)) or len(outputs) != len(
                    states
                ):
                    raise TypeError(
                        "The model must return a sequence of tensors matching the number of modalities."
                    )

                new_states = []
                for idx, config in enumerate(self.modality_configs):
                    current_state = states[idx]
                    model_output = outputs[idx]

                    if config["type"] == "continuous":
                        new_state = current_state + h * model_output
                    elif config["type"] == "discrete":
                        p_1t = model_output
                        dtype = config.get("dtype_categorical", torch.float32)

                        if i == n_steps - 1:
                            new_state = categorical(p_1t.to(dtype))
                        else:
                            path: MixtureDiscreteProbPath = config["path"]
                            x_1 = categorical(p_1t.to(dtype))
                            scheduler_output = path.scheduler(t=t)
                            p_th_t = path.conditional_probability(
                                x_1=x_1,
                                x_t=current_state,
                                t=t,
                                h=h,
                                alpha_t=scheduler_output["alpha_t"],
                                sigma_t=scheduler_output["sigma_t"],
                            )
                            new_state = categorical(p_th_t.to(dtype))

                    new_states.append(new_state)
                states = new_states

                if return_intermediates:
                    for idx, s in enumerate(states):
                        intermediates[idx].append(s.clone())

        return intermediates if return_intermediates else states
