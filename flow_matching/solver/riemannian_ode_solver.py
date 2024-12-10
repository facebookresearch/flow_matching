# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Callable

import torch
from torch import Tensor
from tqdm import tqdm

from flow_matching.solver.solver import Solver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import geodesic, Manifold


class RiemannianODESolver(Solver):
    r"""Riemannian ODE solver
    Initialize the ``RiemannianODESolver``.

    Args:
        manifold (Manifold): the manifold to solve on.
        velocity_model (ModelWrapper): a velocity field model receiving :math:`(x,t)`
            and returning :math:`u_t(x)` which is assumed to lie on the tangent plane at `x`.
    """

    def __init__(self, manifold: Manifold, velocity_model: ModelWrapper):
        super().__init__()
        self.manifold = manifold
        self.velocity_model = velocity_model

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: float,
        projx: bool = True,
        proju: bool = True,
        method: str = "euler",
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        r"""Solve the ODE with the `velocity_field` on the manifold.

        Args:
            x_init (Tensor): initial conditions (e.g., source samples :math:`X_0 \sim p`).
            step_size (float): The step size.
            projx (bool): Whether to project the point onto the manifold at each step. Defaults to True.
            proju (bool): Whether to project the vector field onto the tangent plane at each step. Defaults to True.
            method (str): One of ["euler", "midpoint", "rk4"]. Defaults to "euler".
            time_grid (Tensor, optional): The process is solved in the interval [min(time_grid, max(time_grid)] and if step_size is None then time discretization is set by the time grid. Defaults to torch.tensor([0.0,1.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Defaults to False.
            verbose (bool, optional): Whether to print progress bars. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Tensor: The sampled sequence. Defaults to returning samples at :math:`t=1`.
        """
        step_fns = {
            "euler": _euler_step,
            "midpoint": _midpoint_step,
            "rk4": _rk4_step,
        }
        assert method in step_fns.keys(), f"Unknown method {method}"
        step_fn = step_fns[method]

        # --- Factor this out.
        time_grid = torch.sort(time_grid.to(device=x_init.device)).values

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
            ) > step_size, f"Time interval [min(time_grid), max(time_grid)] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

            n_steps = math.ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )
        # ---
        t0s = t_discretization[:-1]

        if verbose:
            t0s = tqdm(t0s)

        if return_intermediates:
            xts = []
            i_ret = 0

        xt = x_init
        for t0, t1 in zip(t0s, t_discretization[1:]):
            dt = t1 - t0
            xt_next = step_fn(
                self.velocity_model,
                xt,
                t0,
                dt,
                manifold=self.manifold,
                projx=projx,
                proju=proju,
            )
            if return_intermediates:
                while (
                    i_ret < len(time_grid)
                    and t0 <= time_grid[i_ret]
                    and time_grid[i_ret] <= t1
                ):
                    xts.append(
                        interp(self.manifold, xt, xt_next, t0, t1, time_grid[i_ret])
                    )
                    i_ret += 1
            xt = xt_next

        if return_intermediates:
            return torch.stack(xts, dim=0)
        else:
            return xt


def interp(manifold, xt, xt_next, t, t_next, t_ret):
    return geodesic(manifold, xt, xt_next)(
        (t_ret - t) / (t_next - t).reshape(1)
    ).reshape_as(xt)


def _euler_step(
    velocity_model: Callable,
    xt: Tensor,
    t0: Tensor,
    dt: Tensor,
    manifold: Manifold,
    projx: bool = True,
    proju: bool = True,
) -> Tensor:
    r"""Perform an Euler step on a manifold.

    Args:
        velocity_model (Callable): the velocity model
        xt (Tensor): tensor containing the state at time t0
        t0 (Tensor): the time at which this step is taken
        dt (Tensor): the step size
        manifold (Manifold): a manifold object
        projx (bool, optional): whether to project the state onto the manifold. Defaults to True.
        proju (bool, optional): whether to project the velocity onto the tangent plane. Defaults to True.

    Returns:
        Tensor: tensor containing the state after the step
    """
    velocity_fn = lambda x, t: (
        manifold.proju(x, velocity_model(x, t)) if proju else velocity_model(x, t)
    )
    projx_fn = lambda x: manifold.projx(x) if projx else x

    vt = velocity_fn(xt, t0)

    xt = xt + dt * vt

    return projx_fn(xt)


def _midpoint_step(
    velocity_model: Callable,
    xt: Tensor,
    t0: Tensor,
    dt: Tensor,
    manifold: Manifold,
    projx: bool = True,
    proju: bool = True,
) -> Tensor:
    r"""Perform a midpoint step on a manifold.

    Args:
        velocity_model (Callable): the velocity model
        xt (Tensor): tensor containing the state at time t0
        t0 (Tensor): the time at which this step is taken
        dt (Tensor): the step size
        manifold (Manifold): a manifold object
        projx (bool, optional): whether to project the state onto the manifold. Defaults to True.
        proju (bool, optional): whether to project the velocity onto the tangent plane. Defaults to True.

    Returns:
        Tensor: tensor containing the state after the step
    """
    velocity_fn = lambda x, t: (
        manifold.proju(x, velocity_model(x, t)) if proju else velocity_model(x, t)
    )
    projx_fn = lambda x: manifold.projx(x) if projx else x

    half_dt = 0.5 * dt
    vt = velocity_fn(xt, t0)
    x_mid = xt + half_dt * vt
    x_mid = projx_fn(x_mid)

    xt = xt + dt * velocity_fn(x_mid, t0 + half_dt)

    return projx_fn(xt)


def _rk4_step(
    velocity_model: Callable,
    xt: Tensor,
    t0: Tensor,
    dt: Tensor,
    manifold: Manifold,
    projx: bool = True,
    proju: bool = True,
) -> Tensor:
    r"""Perform an RK4 step on a manifold.

    Args:
        velocity_model (Callable): the velocity model
        xt (Tensor): tensor containing the state at time t0
        t0 (Tensor): the time at which this step is taken
        dt (Tensor): the step size
        manifold (Manifold): a manifold object
        projx (bool, optional): whether to project the state onto the manifold. Defaults to True.
        proju (bool, optional): whether to project the velocity onto the tangent plane. Defaults to True.

    Returns:
        Tensor: tensor containing the state after the step
    """
    velocity_fn = lambda x, t: (
        manifold.proju(x, velocity_model(x, t)) if proju else velocity_model(x, t)
    )
    projx_fn = lambda x: manifold.projx(x) if projx else x

    k1 = velocity_fn(xt, t0)
    k2 = velocity_fn(projx_fn(xt + dt * k1 / 3), t0 + dt / 3)
    k3 = velocity_fn(projx_fn(xt + dt * (k2 - k1 / 3)), t0 + dt * 2 / 3)
    k4 = velocity_fn(projx_fn(xt + dt * (k1 - k2 + k3)), t0 + dt)

    return projx_fn(xt + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125)
