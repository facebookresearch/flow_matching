# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from flow_matching.utils import expand_tensor_like, gradient, unsqueeze_to_match


class TestUtils(unittest.TestCase):
    def test_unsqueeze_to_match_suffix(self):
        source = torch.randn(3)
        target = torch.randn(3, 4, 5)
        result = unsqueeze_to_match(source, target)
        self.assertEqual(result.shape, (3, 1, 1))

    def test_unsqueeze_to_match_prefix(self):
        source = torch.randn(3)
        target = torch.randn(4, 5, 3)
        result = unsqueeze_to_match(source, target, how="prefix")
        self.assertEqual(result.shape, (1, 1, 3))

    def test_expand_tensor_like(self):
        input_tensor = torch.randn(3)
        expand_to = torch.randn(3, 4, 5)
        result = expand_tensor_like(input_tensor, expand_to)
        self.assertEqual(result.shape, (3, 4, 5))

    def test_gradient(self):
        x = torch.randn(3, requires_grad=True)
        output = x**2
        grad_outputs = torch.ones_like(output)
        result = gradient(output, x, grad_outputs=grad_outputs)
        self.assertTrue(torch.allclose(result, 2 * x))


if __name__ == "__main__":
    unittest.main()
