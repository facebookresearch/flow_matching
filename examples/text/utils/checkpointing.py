# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

from dataclasses import dataclass, field
from pathlib import Path

import torch
from logic.flow import SourceDistribution
from model import Transformer
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def load_cfg_from_path(work_dir: str) -> OmegaConf:
    work_dir = Path(work_dir)

    root_dir = work_dir if work_dir.is_dir() else work_dir.parents[1]

    cfg_path = root_dir / ".hydra/config.yaml"

    return OmegaConf.load(cfg_path)


def load_model_from_path(
    work_dir: str,
    source_distribution: SourceDistribution,
    device: torch.device,
    vocab_size: int,
    cfg: OmegaConf,
) -> nn.Module:
    work_dir = Path(work_dir)

    if work_dir.is_dir():
        root_dir = work_dir
        ckpt_dir = work_dir / "checkpoints" / "checkpoint.pth"
    else:
        root_dir = work_dir.parents[1]
        ckpt_dir = work_dir

    model = Transformer(
        config=cfg, vocab_size=vocab_size, masked=source_distribution.masked
    ).to(device)
    model = DDP(model, device_ids=[device])

    ckpt_dir = root_dir / "checkpoints" / "checkpoint.pth"
    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=True)

    model.module.load_state_dict(loaded_state["model"])

    return model


@dataclass
class WorkDirectory:
    root: Path = field(metadata={"help": "Root work directory"})
    checkpoint: Path = field(metadata={"help": "Checkpoint directory"})
    samples: Path = field(metadata={"help": "Samples directory"})


def get_work_dirs(work_dir: str, rank: int) -> WorkDirectory:
    work_dir = Path(work_dir)

    sample_dir = work_dir / "samples"
    checkpoint_dir = work_dir / "checkpoints" / "checkpoint.pth"

    if rank == 0:
        sample_dir.mkdir(exist_ok=True)
        checkpoint_dir.parents[0].mkdir(exist_ok=True)

    return WorkDirectory(root=work_dir, checkpoint=checkpoint_dir, samples=sample_dir)
