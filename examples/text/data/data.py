# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Iterable, Tuple

from datasets import DatasetDict, load_dataset
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

from data.tokenizer import wt_detokenizer
from data.utils import cycle_loader, StatefulDistributedSampler


def _get_hf_dataset(
    name: str,
    mode: str,
    cache_dir: str = None,
    block_size: int = 1024,
    num_proc: int = 8,
) -> DatasetDict:
    detokenizer = None

    if name == "wikitext103":
        data = load_dataset(
            "wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir
        )[mode]
        detokenizer = wt_detokenizer
    elif name == "fineweb-edu":
        data = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", cache_dir=cache_dir
        )[mode]
    else:
        data = load_dataset(name, cache_dir=cache_dir)[mode]

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text

        return detok

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example: Dict):
        text = example["text"]

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token following
        # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        for token in tokens["input_ids"]:
            token.append(EOS)

        return tokens

    tokenized_dataset = data.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
    )

    if name == "fineweb-edu":
        features = tokenized_dataset.features.keys()
        for k in features:
            if k != "input_ids":
                tokenized_dataset = tokenized_dataset.remove_columns(k)
    else:
        tokenized_dataset = tokenized_dataset.remove_columns("text")

    def group_texts(examples: Dict):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        return result

    chunked_dataset = tokenized_dataset.map(
        group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True
    )
    chunked_dataset = chunked_dataset.with_format("torch")

    return chunked_dataset


@dataclass
class Dataset:
    dataset: DatasetDict = field(metadata={"help": "Huggingface dataset"})
    sampler: StatefulDistributedSampler = field(
        metadata={"help": "Stateful sampler for `dataset`"}
    )


@dataclass
class DataState:
    train: Dataset = field(metadata={"help": "Train dataset"})
    test: Dataset = field(metadata={"help": "Test dataset"})


def _get_dataset(
    name: str,
    mode: str,
    cache_dir: str,
    block_size: int,
    num_proc: int,
    batch_size: int,
    ngpus: int,
) -> Dataset:
    assert (
        batch_size % ngpus == 0
    ), f"{mode} batch size must be divisible by number of gpus."

    dataset = _get_hf_dataset(
        name=name,
        mode=mode,
        cache_dir=cache_dir,
        block_size=block_size,
        num_proc=num_proc,
    )

    sampler = StatefulDistributedSampler(dataset=dataset)

    return Dataset(dataset=dataset, sampler=sampler)


def get_data_state(config: OmegaConf) -> DataState:
    train = _get_dataset(
        name=config.data.train,
        mode="train",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.training.batch_size,
        ngpus=config.compute.ngpus,
    )
    test = _get_dataset(
        name=config.data.valid,
        mode="validation",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.eval.batch_size,
        ngpus=config.compute.ngpus,
    )

    return DataState(train=train, test=test)


def get_data_loaders(
    config: OmegaConf,
    data_state: DataState,
) -> Tuple[Iterable, Iterable]:
    train_loader = cycle_loader(
        DataLoader(
            data_state.train.dataset,
            batch_size=config.training.batch_size // config.compute.ngpus,
            sampler=data_state.train.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.train.sampler is None),
            persistent_workers=True,
        )
    )

    valid_loader = cycle_loader(
        DataLoader(
            data_state.test.dataset,
            batch_size=config.eval.batch_size // config.compute.ngpus,
            sampler=data_state.test.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.test.sampler is None),
        )
    )

    return iter(train_loader), iter(valid_loader)
