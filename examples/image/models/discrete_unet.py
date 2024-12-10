# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
from models.unet import UNetModel


class PixelEmbedding(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        hidden_size: int,
    ):
        super().__init__()
        self.embedding_table = nn.Embedding(n_tokens, hidden_size)

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        emb = self.embedding_table(x)
        result = emb.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        return result


@dataclass(eq=False)
class DiscreteUNetModel(nn.Module):
    vocab_size: int
    in_channels: int = 3
    model_channels: int = 128
    out_channels: int = 3
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int] = (1, 2, 2, 2)
    dropout: float = 0.0
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: Optional[int] = None
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    with_fourier_features: bool = False

    def __post_init__(self):
        super().__init__()
        assert (
            self.model_channels * self.channel_mult[0] % self.in_channels == 0
        ), f"Unet input dimensions must be divisible by the number of channels. Got {self.model_channels * self.channel_mult[0]} / {self.in_channels}"
        self.embedding_dim = (
            self.model_channels * self.channel_mult[0] // self.in_channels
        )

        self.pixel_embedding = PixelEmbedding(
            n_tokens=self.vocab_size, hidden_size=self.embedding_dim
        )

        self.unet = UNetModel(
            in_channels=self.in_channels * self.embedding_dim,
            model_channels=self.model_channels,
            out_channels=self.out_channels * (self.vocab_size),
            num_res_blocks=self.num_res_blocks,
            attention_resolutions=self.attention_resolutions,
            dropout=self.dropout,
            channel_mult=self.channel_mult,
            conv_resample=self.conv_resample,
            dims=self.dims,
            num_classes=self.num_classes,
            use_checkpoint=self.use_checkpoint,
            num_heads=self.num_heads,
            num_head_channels=self.num_head_channels,
            num_heads_upsample=self.num_heads_upsample,
            use_scale_shift_norm=self.use_scale_shift_norm,
            resblock_updown=self.resblock_updown,
            use_new_attention_order=self.use_new_attention_order,
            with_fourier_features=self.with_fourier_features,
            ignore_time=True,
            input_projection=False,
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, extra: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        B, C, H, W = x_t.shape
        logits = (
            self.unet(self.pixel_embedding(x_t), t, extra)
            .reshape(B, C, self.vocab_size, H, W)
            .permute(0, 1, 3, 4, 2)
        )
        return logits
