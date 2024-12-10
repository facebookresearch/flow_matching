# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor

from flow_matching.path.scheduler.scheduler import Scheduler
from flow_matching.utils import ModelWrapper


class ScheduleTransformedModel(ModelWrapper):
    """
    Change of scheduler for a velocity model.

    This class wraps a given velocity model and transforms its scheduling
    to a new scheduler function. It modifies the time
    dynamics of the model according to the new scheduler while maintaining
    the original model's behavior.

    Example:

    .. code-block:: python

        import torch
        from flow_matching.path.scheduler import CondOTScheduler, CosineScheduler, ScheduleTransformedModel
        from flow_matching.solver import ODESolver

        # Initialize the model and schedulers
        model = ...

        original_scheduler = CondOTScheduler()
        new_scheduler = CosineScheduler()

        # Create the transformed model
        transformed_model = ScheduleTransformedModel(
            velocity_model=model,
            original_scheduler=original_scheduler,
            new_scheduler=new_scheduler
        )

        # Set up the solver
        solver = ODESolver(velocity_model=transformed_model)

        x_0 = torch.randn([10, 2])  # Example initial condition

        x_1 = solver.sample(
            time_steps=torch.tensor([0.0, 1.0]),
            x_init=x_0,
            step_size=1/1000
            )[1]

    Args:
        velocity_model (ModelWrapper): The original velocity model to be transformed.
        original_scheduler (Scheduler): The scheduler used by the original model. Must implement the snr_inverse function.
        new_scheduler (Scheduler): The new scheduler to be applied to the model.
    """

    def __init__(
        self,
        velocity_model: ModelWrapper,
        original_scheduler: Scheduler,
        new_scheduler: Scheduler,
    ):
        super().__init__(model=velocity_model)
        self.original_scheduler = original_scheduler
        self.new_scheduler = new_scheduler

        assert hasattr(self.original_scheduler, "snr_inverse") and callable(
            getattr(self.original_scheduler, "snr_inverse")
        ), "The original scheduler must have a callable 'snr_inverse' method."

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        r"""
        Compute the transformed marginal velocity field for a new scheduler.
        This method implements a post-training velocity scheduler change for
        affine conditional flows. It transforms a generating marginal velocity
        field :math:`u_t(x)` based on an original scheduler to a new marginal velocity
        field :math:`\bar{u}_r(x)` based on a different scheduler, while maintaining
        the same data coupling.
        The transformation is based on the scale-time (ST) transformation
        between the two conditional flows, defined as:

        .. math::

            \bar{X}_r = s_r X_{t_r},

        where :math:`X_t` and :math:`\bar{X}_r` are defined by their respective schedulers.
        The ST transformation is computed as:

        .. math::

            t_r = \rho^{-1}(\bar{\rho}(r)) \quad \text{and} \quad  s_r = \frac{\bar{\sigma}_r}{\sigma_{t_r}}.

        Here, :math:`\rho(t)` is the signal-to-noise ratio (SNR) defined as:

        .. math::

            \rho(t) = \frac{\alpha_t}{\sigma_t}.

        :math:`\bar{\rho}(r)` is similarly defined for the new scheduler.
        The marginal velocity for the new scheduler is then given by:

        .. math::

            \bar{u}_r(x) = \left(\frac{\dot{s}_r}{s_r}\right) x + s_r \dot{t}_r u_{t_r}\left(\frac{x}{s_r}\right).

        Args:
            x (Tensor): :math:`x_t`, the input tensor.
            t (Tensor): The time tensor (denoted as :math:`r` above).
            **extras: Additional arguments for the model.
        Returns:
            Tensor: The transformed velocity.
        """
        r = t

        r_scheduler_output = self.new_scheduler(t=r)

        alpha_r = r_scheduler_output.alpha_t
        sigma_r = r_scheduler_output.sigma_t
        d_alpha_r = r_scheduler_output.d_alpha_t
        d_sigma_r = r_scheduler_output.d_sigma_t

        t = self.original_scheduler.snr_inverse(alpha_r / sigma_r)

        t_scheduler_output = self.original_scheduler(t=t)

        alpha_t = t_scheduler_output.alpha_t
        sigma_t = t_scheduler_output.sigma_t
        d_alpha_t = t_scheduler_output.d_alpha_t
        d_sigma_t = t_scheduler_output.d_sigma_t

        s_r = sigma_r / sigma_t

        dt_r = (
            sigma_t
            * sigma_t
            * (sigma_r * d_alpha_r - alpha_r * d_sigma_r)
            / (sigma_r * sigma_r * (sigma_t * d_alpha_t - alpha_t * d_sigma_t))
        )

        ds_r = (sigma_t * d_sigma_r - sigma_r * d_sigma_t * dt_r) / (sigma_t * sigma_t)

        u_t = self.model(x=x / s_r, t=t, **extras)
        u_r = ds_r * x / s_r + dt_r * s_r * u_t

        return u_r
