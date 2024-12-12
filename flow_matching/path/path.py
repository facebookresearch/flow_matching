# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

from torch import Tensor

from flow_matching.path.path_sample import PathSample


class ProbPath(ABC):
    r"""Abstract class, representing a probability path.

    A probability path transforms the distribution :math:`p(X_0)` into :math:`p(X_1)` over :math:`t=0\rightarrow 1`.

    The ``ProbPath`` class is designed to support model training in the flow matching framework. It supports two key functionalities: (1) sampling the conditional probability path and (2) conversion between various training objectives.
    Here is a high-level example

    .. code-block:: python

        # Instantiate a probability path
        my_path = ProbPath(...)

        for x_0, x_1 in dataset:
            # Sets t to a random value in [0,1]
            t = torch.rand()

            # Samples the conditional path X_t ~ p_t(X_t|X_0,X_1)
            path_sample = my_path.sample(x_0=x_0, x_1=x_1, t=t)

            # Optimizes the model. The loss function varies, depending on model and path.
            loss(path_sample, my_model(x_t, t)).backward()

    """

    @abstractmethod
    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        r"""Sample from an abstract probability path:

        | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)`.
        | returns :math:`X_0, X_1, X_t \sim p_t(X_t)`, and a conditional target :math:`Y`, all objects are under ``PathSample``.

        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            PathSample: a conditional sample.
        """

    def assert_sample_shape(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> None:
        """Assert that the shapes of input tensors are compatible.
        
        Args:
            x_0 (Tensor): source data point, shape (batch_size, ..., feature_dim)
            x_1 (Tensor): target data point, shape (batch_size, ..., feature_dim)
            t (Tensor): times in [0,1], shape (batch_size, ...)
        """
        # Check that x_0 and x_1 have the same shape
        assert x_0.shape == x_1.shape, f"x_0 shape {x_0.shape} must match x_1 shape {x_1.shape}"
        
        # Check that t has one dimension less than x_0/x_1
        assert len(t.shape) == len(x_0.shape) - 1, f"t shape {t.shape} must have one dimension less than x_0/x_1 shape {x_0.shape}"
        
        # Check that all but the last dimension match between t and x_0/x_1
        assert t.shape == x_0.shape[:-1], f"t shape {t.shape} must match all but last dimension of x_0/x_1 shape {x_0.shape}"
