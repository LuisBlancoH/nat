"""
Data loading and episode construction for NAT training.

Provides:
  - ``build_phase1_dataloader``  — diverse text for meta-learning θ.
  - ``build_phase2_dataloader``  — episodic multi-task data for Phase 2.
  - ``build_text_dataset``       — generic tokenised-chunked dataset.
  - ``EpisodeDataset``           — wraps an iterable HF dataset into
    fixed-length token sequences suitable for episodic training.

Supported data sources
----------------------
- HuggingFace ``datasets`` streaming corpora (C4, SlimPajama, etc.)
- Local text / JSONL files
- A lightweight *synthetic* dataset for unit-testing and development
  (``SyntheticEpisodeDataset``).
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Synthetic dataset (for tests and small-scale debugging)              #
# ------------------------------------------------------------------ #

class SyntheticEpisodeDataset(Dataset):
    """
    Deterministic dataset of random token sequences.

    Useful for:
      - Unit tests that need a real ``DataLoader`` but should not
        download anything.
      - Fast smoke-tests on local hardware.
      - Verifying the training loop mechanics independently of data.

    Each item is a dict ``{"input_ids": LongTensor(seq_len)}``.
    """

    def __init__(
        self,
        num_episodes: int = 256,
        seq_len: int = 256,
        vocab_size: int = 1000,
        seed: int = 42,
    ):
        super().__init__()
        self.num_episodes = num_episodes
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed
        # Pre-generate all episodes for determinism
        rng = torch.Generator().manual_seed(seed)
        self.data = [
            torch.randint(0, vocab_size, (seq_len,), generator=rng)
            for _ in range(num_episodes)
        ]

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.data[idx]}


# ------------------------------------------------------------------ #
# Tokenised-and-chunked wrapper for HF streaming datasets              #
# ------------------------------------------------------------------ #

class TokenChunkedDataset(IterableDataset):
    """
    Streams text from a HuggingFace dataset, tokenises on the fly,
    and yields fixed-length chunks of ``seq_len`` tokens.

    This is an *infinite* iterable dataset (wraps around).  Use it
    with a ``DataLoader(dataset, batch_size=...)`` and stop after
    ``num_episodes`` batches.

    Parameters
    ----------
    hf_dataset
        A HuggingFace ``IterableDataset`` (``streaming=True``).
    tokenizer
        A HuggingFace tokenizer.
    seq_len : int
        Number of tokens per chunk.
    text_column : str
        Name of the text column in the HF dataset.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        seq_len: int = 2048,
        text_column: str = "text",
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []

        for example in self.hf_dataset:
            text = example[self.text_column]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}


# ------------------------------------------------------------------ #
# Dataloader builders                                                  #
# ------------------------------------------------------------------ #

def build_phase1_dataloader(
    config,
    tokenizer=None,
    *,
    synthetic: bool = False,
) -> DataLoader:
    """
    Build a ``DataLoader`` for Phase 1 meta-learning.

    Parameters
    ----------
    config : NATConfig
        Must have ``seq_len``, ``batch_size``, ``num_episodes_p1``.
    tokenizer : optional
        HuggingFace tokenizer.  Required unless ``synthetic=True``.
    synthetic : bool
        If ``True``, use ``SyntheticEpisodeDataset`` instead of real data.
        Intended for tests and fast iteration.

    Returns
    -------
    DataLoader
    """
    if synthetic:
        dataset = SyntheticEpisodeDataset(
            num_episodes=config.num_episodes_p1,
            seq_len=config.seq_len,
            vocab_size=getattr(config, "vocab_size", 1000),
        )
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

    # ---- Real data ----
    assert tokenizer is not None, (
        "tokenizer is required for non-synthetic data. "
        "Pass synthetic=True for testing."
    )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install HuggingFace datasets: pip install datasets"
        ) from exc

    # Load a diverse text corpus via streaming
    # Default: C4 (en) — widely available, diverse, large.
    dataset_name = getattr(config, "dataset_name", "allenai/c4")
    dataset_config = getattr(config, "dataset_config", "en")
    text_column = getattr(config, "text_column", "text")

    logger.info(
        f"Loading streaming dataset: {dataset_name} ({dataset_config})"
    )

    hf_ds = load_dataset(
        dataset_name,
        dataset_config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    chunked = TokenChunkedDataset(
        hf_dataset=hf_ds,
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        text_column=text_column,
    )

    return DataLoader(
        chunked,
        batch_size=config.batch_size,
        num_workers=0,  # streaming + tokenisation is fast enough in main
    )


def collate_episodes(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Default collate that stacks ``input_ids`` tensors.

    Parameters
    ----------
    batch : list[dict]
        Each dict has at least ``{"input_ids": LongTensor(seq_len)}``.

    Returns
    -------
    dict with ``"input_ids"`` of shape ``(batch_size, seq_len)``.
    """
    return {"input_ids": torch.stack([b["input_ids"] for b in batch])}


# ------------------------------------------------------------------ #
# Synthetic episodic dataset (structured problem/solution sequences)   #
# ------------------------------------------------------------------ #

class SyntheticEpisodicDataset(Dataset):
    """
    Generates synthetic episodic data with problem/solution structure.

    Each episode is a flat token sequence divided into ``num_problems``
    equal-sized "problems".  Each problem has a "problem region" and a
    "solution region" (the second half of each problem block).

    Returns ``input_ids`` and ``problem_spans`` — a list of
    ``(sol_start, sol_end)`` index pairs identifying where each
    solution region sits in the sequence.

    This allows testing the episodic training loop (including the
    improvement bonus) without real data.
    """

    def __init__(
        self,
        num_episodes: int = 256,
        seq_len: int = 256,
        num_problems: int = 4,
        vocab_size: int = 1000,
        seed: int = 42,
    ):
        super().__init__()
        self.num_episodes = num_episodes
        self.seq_len = seq_len
        self.num_problems = num_problems
        self.vocab_size = vocab_size

        rng = torch.Generator().manual_seed(seed)
        self.data: list[dict[str, Any]] = []

        # Pre-compute problem span boundaries (same for all episodes)
        tokens_per_problem = seq_len // num_problems
        self.problem_spans: list[tuple[int, int]] = []
        for i in range(num_problems):
            block_start = i * tokens_per_problem
            sol_start = block_start + tokens_per_problem // 2
            sol_end = block_start + tokens_per_problem
            # Ensure sol_start >= 1 (we need logits at sol_start-1)
            sol_start = max(sol_start, 1)
            self.problem_spans.append((sol_start, sol_end))

        for _ in range(num_episodes):
            ids = torch.randint(0, vocab_size, (seq_len,), generator=rng)
            self.data.append(ids)

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "input_ids": self.data[idx],
            "problem_spans": self.problem_spans,
        }


def collate_episodic(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for episodic datasets.

    Stacks ``input_ids`` and uses the problem_spans from the first
    item (they're identical across all items in a batch of
    ``SyntheticEpisodicDataset``).
    """
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "problem_spans": batch[0]["problem_spans"],
    }


def build_phase2_dataloader(
    config,
    tokenizer=None,
    *,
    synthetic: bool = False,
) -> DataLoader:
    """
    Build a ``DataLoader`` for Phase 2 episodic training.

    Parameters
    ----------
    config : NATConfig
        Must have ``seq_len``, ``batch_size``, ``num_episodes_p2``,
        ``num_problems_per_episode``.
    tokenizer : optional
        HuggingFace tokenizer.  Required unless ``synthetic=True``.
    synthetic : bool
        If ``True``, use ``SyntheticEpisodicDataset``.

    Returns
    -------
    DataLoader
    """
    num_problems = getattr(config, "num_problems_per_episode", 8)

    if synthetic:
        dataset = SyntheticEpisodicDataset(
            num_episodes=getattr(config, "num_episodes_p2", 1000),
            seq_len=config.seq_len,
            num_problems=num_problems,
            vocab_size=getattr(config, "vocab_size", 1000),
        )
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            collate_fn=collate_episodic,
        )

    # ---- Real episodic data ----
    # For real data, we'd load task-specific datasets (GSM8K, HumanEval, etc.),
    # format as problem/solution pairs, tokenise, and record spans.
    # This is highly task-dependent — the synthetic path above covers
    # the training loop mechanics; real data wiring is left to the user.
    assert tokenizer is not None, (
        "tokenizer is required for non-synthetic data. "
        "Pass synthetic=True for testing."
    )
    raise NotImplementedError(
        "Real episodic data loading is task-dependent. "
        "Subclass SyntheticEpisodicDataset or provide your own DataLoader."
    )
