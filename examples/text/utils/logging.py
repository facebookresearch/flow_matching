# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from logging import Logger
from pathlib import Path
from typing import Optional

import torch
import wandb
from omegaconf import OmegaConf


def get_logger(log_path: str, rank: int):
    if rank != 0:
        return logging.getLogger("dummy")

    logger = logging.getLogger()
    default_level = logging.INFO

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(default_level)

    formatter = logging.Formatter(
        "%(levelname)s | %(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    info_file_handler = logging.FileHandler(log_path, mode="a")
    info_file_handler.setLevel(default_level)
    info_file_handler.setFormatter(formatter)
    logger.addHandler(info_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(default_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class TrainLogger:
    def __init__(self, log_dir: Path, rank: int, cfg: bool = False):
        self.log_dir = log_dir
        self.cfg = cfg

        self._init_text_logger(rank=rank)

        self.enable_wandb = self.cfg.logging.enable_wandb and (rank == 0)

        if self.enable_wandb:
            self._init_wandb()

    def _init_text_logger(self, rank: int):
        log_path = self.log_dir / self.cfg.logging.log_file_name
        self._logger = get_logger(log_path=log_path, rank=rank)

    def _init_wandb(
        self,
    ):
        wandb_run_id_path = self.log_dir / "wandb_run.id"

        try:
            wandb_run_id = wandb_run_id_path.read_text()
        except FileNotFoundError:
            wandb_run_id = wandb.util.generate_id()
            wandb_run_id_path.write_text(wandb_run_id)

        self.wandb_logger = wandb.init(
            id=wandb_run_id,
            project=self.cfg.logging.project,
            group=self.cfg.logging.group,
            dir=self.log_dir,
            entity=self.cfg.logging.entity,
            resume="allow",
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )

    def log_metric(self, value: float, name: str, stage: bool, step: int) -> None:
        self._logger.info(f"[{step}] {stage} {name}: {value:.3f}")

        if self.enable_wandb:
            self.wandb_logger.log(data={f"{stage}/{name}": value}, step=step)

    def log_lr(self, value: float, step: int) -> None:
        if self.enable_wandb:
            self.wandb_logger.log(data={"Optimization/LR": value}, step=step)

    def info(self, msg: str, step: Optional[int] = None) -> None:
        step_str = f"[{step}] " if step else ""
        self._logger.info(f"{step_str}{msg}")

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def finish(self) -> None:
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()

        if self.enable_wandb:
            wandb.finish()

    @staticmethod
    def log_devices(device: torch.device, logger: Logger) -> None:
        if device.type == "cuda":
            logger.info("Found {} CUDA devices.".format(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(
                    "{} \t Memory: {:.2f}GB".format(
                        props.name, props.total_memory / (1024**3)
                    )
                )
        else:
            logger.warning("WARNING: Using device {}".format(device))
        logger.info(f"Found {os.cpu_count()} total number of CPUs.")
