"""Streaming text dataset → fixed-length token windows.

Yields contiguous chunks of `seq_len + 1` token ids so the training loop
can build (input, target) pairs for next-token prediction. Streams from
HuggingFace Datasets so we never pay for full download.
"""
from __future__ import annotations

import itertools
from typing import Iterator

import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast


def get_tokenizer() -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _token_stream(
    dataset_name: str,
    split: str,
    text_field: str,
    tokenizer: GPT2TokenizerFast,
    seed: int,
    skip: int = 0,
) -> Iterator[int]:
    """Yield one token id at a time, indefinitely."""
    ds = load_dataset(dataset_name, split=split, streaming=True)
    ds = ds.shuffle(buffer_size=1000, seed=seed)
    eos = tokenizer.eos_token_id
    for i, ex in enumerate(ds):
        if i < skip:
            continue
        text = ex.get(text_field) or ""
        if not text:
            continue
        ids = tokenizer.encode(text)
        yield from ids
        yield eos


def windowed_batches(
    dataset_name: str,
    split: str,
    text_field: str,
    tokenizer: GPT2TokenizerFast,
    seq_len: int,
    batch_size: int,
    seed: int = 0,
    skip: int = 0,
) -> Iterator[torch.Tensor]:
    """Yield int64 tensors of shape (batch_size, seq_len + 1)."""
    stream = _token_stream(dataset_name, split, text_field, tokenizer, seed, skip)
    window = seq_len + 1
    needed = batch_size * window
    while True:
        chunk = list(itertools.islice(stream, needed))
        if len(chunk) < needed:
            return  # stream exhausted (split too small)
        t = torch.tensor(chunk, dtype=torch.long).view(batch_size, window)
        yield t
