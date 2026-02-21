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
    item (they're identical across all items in a batch because both
    ``SyntheticEpisodicDataset`` and ``RealEpisodicDataset`` use
    fixed-size slots with uniform span layout).
    """
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "problem_spans": batch[0]["problem_spans"],
    }


# ------------------------------------------------------------------ #
# Real episodic dataset (multi-task QA from HuggingFace)               #
# ------------------------------------------------------------------ #

class RealEpisodicDataset(Dataset):
    """
    Episodic dataset built from real QA datasets.

    Loads questions from multiple HuggingFace datasets, formats them as
    ``Question: ...\\nAnswer: ...\\n\\n`` pairs, packs ``num_problems``
    per episode into a single tokenised sequence, and records
    ``problem_spans`` for per-problem loss computation.

    Sources (~380 K QA pairs total, each loaded gracefully):
      - ``openai/gsm8k``        — grade-school math (~7.5 K)
      - ``allenai/ai2_arc``     — science reasoning, Easy + Challenge (~3.4 K)
      - ``cais/mmlu``           — multi-domain knowledge (~14 K)
      - ``tau/commonsense_qa``  — commonsense reasoning (~10 K)
      - ``ybisk/piqa``          — physical intuition (~16 K)
      - ``google/boolq``        — yes / no reading comprehension (~9.4 K)
      - ``Rowan/hellaswag``     — commonsense completion (~40 K)
      - ``allenai/sciq``        — science with explanations (~12 K)
      - ``allenai/openbookqa``  — open-book science QA (~5 K)
      - ``trivia_qa``           — trivia / factual recall (~138 K)
      - ``allenai/winogrande``  — coreference / commonsense (~40 K)

    Falls back gracefully if a dataset is unavailable.
    """

    SOURCES = [
        # ── Math ──
        {
            "name": "openai/gsm8k",
            "config": "main",
            "split": "train",
            "formatter": lambda ex: (
                ex["question"],
                ex["answer"].split("####")[-1].strip()
                if "####" in ex["answer"]
                else ex["answer"],
            ),
        },
        # ── Science (easy) ──
        {
            "name": "allenai/ai2_arc",
            "config": "ARC-Easy",
            "split": "train",
            "formatter": lambda ex: (
                ex["question"],
                ex["choices"]["text"][
                    ex["choices"]["label"].index(ex["answerKey"])
                ]
                if ex["answerKey"] in ex["choices"]["label"]
                else ex["choices"]["text"][0],
            ),
        },
        # ── Science (hard) ──
        {
            "name": "allenai/ai2_arc",
            "config": "ARC-Challenge",
            "split": "train",
            "formatter": lambda ex: (
                ex["question"],
                ex["choices"]["text"][
                    ex["choices"]["label"].index(ex["answerKey"])
                ]
                if ex["answerKey"] in ex["choices"]["label"]
                else ex["choices"]["text"][0],
            ),
        },
        # ── Multi-domain knowledge ──
        {
            "name": "cais/mmlu",
            "config": "all",
            "split": "test",
            "formatter": lambda ex: (
                ex["question"],
                ex["choices"][ex["answer"]]
                if isinstance(ex["answer"], int)
                and 0 <= ex["answer"] < len(ex["choices"])
                else ex["choices"][0],
            ),
        },
        # ── Commonsense reasoning ──
        {
            "name": "tau/commonsense_qa",
            "config": None,
            "split": "train",
            "formatter": lambda ex: (
                ex["question"],
                ex["choices"]["text"][
                    ex["choices"]["label"].index(ex["answerKey"])
                ]
                if ex["answerKey"] in ex["choices"]["label"]
                else ex["choices"]["text"][0],
            ),
        },
        # ── Physical intuition ──
        {
            "name": "ybisk/piqa",
            "config": None,
            "split": "train",
            "formatter": lambda ex: (
                ex["goal"],
                ex["sol1"] if ex["label"] == 0 else ex["sol2"],
            ),
        },
        # ── Reading comprehension (yes/no) ──
        {
            "name": "google/boolq",
            "config": None,
            "split": "train",
            "formatter": lambda ex: (
                ex["question"],
                "Yes" if ex["answer"] else "No",
            ),
        },
        # ── Commonsense completion ──
        {
            "name": "Rowan/hellaswag",
            "config": None,
            "split": "train",
            "formatter": lambda ex: (
                ex["ctx"],
                ex["endings"][int(ex["label"])]
                if str(ex["label"]).isdigit()
                else ex["endings"][0],
            ),
        },
        # ── Science with explanations ──
        {
            "name": "allenai/sciq",
            "config": None,
            "split": "train",
            "formatter": lambda ex: (
                ex["question"],
                ex["correct_answer"],
            ),
        },
        # ── Open-book science QA ──
        {
            "name": "allenai/openbookqa",
            "config": "main",
            "split": "train",
            "formatter": lambda ex: (
                ex["question_stem"],
                ex["choices"]["text"][
                    ex["choices"]["label"].index(ex["answerKey"])
                ]
                if ex["answerKey"] in ex["choices"]["label"]
                else ex["choices"]["text"][0],
            ),
        },
        # ── Trivia / factual recall ──
        {
            "name": "trivia_qa",
            "config": "rc.nocontext",
            "split": "train",
            "formatter": lambda ex: (
                ex["question"],
                ex["answer"]["value"]
                if ex["answer"].get("value")
                else (
                    ex["answer"]["aliases"][0]
                    if ex["answer"].get("aliases")
                    else ""
                ),
            ),
        },
        # ── Coreference / commonsense ──
        {
            "name": "allenai/winogrande",
            "config": "winogrande_xl",
            "split": "train",
            "formatter": lambda ex: (
                ex["sentence"],
                ex["option1"] if ex["answer"] == "1" else ex["option2"],
            ),
        },
    ]

    def __init__(
        self,
        tokenizer,
        num_episodes: int = 30000,
        seq_len: int = 2048,
        num_problems: int = 8,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_episodes = num_episodes
        self.seq_len = seq_len
        self.num_problems = num_problems

        # Load QA pairs grouped by source for balanced sampling
        self.source_buckets: list[list[tuple[str, str]]] = []
        self._load_sources()

        if not self.source_buckets:
            raise RuntimeError(
                "No QA pairs loaded. Check network connectivity and "
                "dataset availability."
            )

        # Pre-compute fixed problem spans (identical for every episode)
        tokens_per_problem = seq_len // num_problems
        self.problem_spans: list[tuple[int, int]] = []
        for i in range(num_problems):
            block_start = i * tokens_per_problem
            sol_start = block_start + tokens_per_problem // 2
            sol_end = block_start + tokens_per_problem
            sol_start = max(sol_start, 1)
            self.problem_spans.append((sol_start, sol_end))

        total = sum(len(b) for b in self.source_buckets)
        logger.info(
            f"Loaded {total} QA pairs across "
            f"{len(self.source_buckets)} sources for Phase 2 "
            f"(source-balanced sampling)"
        )

    def _load_sources(self):
        """Load QA pairs from HuggingFace datasets into per-source buckets."""
        from datasets import load_dataset

        for src in self.SOURCES:
            bucket: list[tuple[str, str]] = []
            try:
                logger.info(f"Loading {src['name']} ({src['config']})...")
                ds = load_dataset(
                    src["name"],
                    src["config"],
                    split=src["split"],
                )
                for example in ds:
                    try:
                        q, a = src["formatter"](example)
                        if q and a:
                            bucket.append((q.strip(), a.strip()))
                    except (KeyError, IndexError, TypeError):
                        continue
                if bucket:
                    self.source_buckets.append(bucket)
                    logger.info(
                        f"  → {src['name']}: {len(bucket)} pairs"
                    )
                else:
                    logger.warning(
                        f"  → {src['name']}: loaded but 0 valid pairs"
                    )
            except Exception as e:
                logger.warning(f"  → {src['name']}: failed ({e}), skipping")

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Build one episode: num_problems QA pairs in fixed-size slots.

        Each slot is ``seq_len // num_problems`` tokens.  The first half
        of the slot holds the question tokens, the second half holds the
        answer tokens, mirroring the layout used by
        ``SyntheticEpisodicDataset`` so that ``collate_episodic`` can
        use a single ``problem_spans`` for the whole batch.
        """
        rng = random.Random(idx)

        slot_size = self.seq_len // self.num_problems
        q_budget = slot_size // 2          # first half → question
        a_budget = slot_size - q_budget    # second half → answer
        pad_id = self.tokenizer.eos_token_id or 0

        all_tokens: list[int] = []

        # Source-balanced sampling: pick a random source, then a
        # random example from it — every source gets equal weight
        # regardless of its size.
        selected = [
            rng.choice(rng.choice(self.source_buckets))
            for _ in range(self.num_problems)
        ]

        for q, a in selected:
            q_tokens = self.tokenizer.encode(
                f"Question: {q}\nAnswer: ", add_special_tokens=False
            )[:q_budget]
            a_tokens = self.tokenizer.encode(
                f"{a}\n\n", add_special_tokens=False
            )[:a_budget]

            # Pad question half, then answer half
            slot = (
                q_tokens + [pad_id] * (q_budget - len(q_tokens))
                + a_tokens + [pad_id] * (a_budget - len(a_tokens))
            )
            all_tokens.extend(slot)

        # Handle rounding remainder
        if len(all_tokens) < self.seq_len:
            all_tokens.extend([pad_id] * (self.seq_len - len(all_tokens)))
        all_tokens = all_tokens[:self.seq_len]

        return {
            "input_ids": torch.tensor(all_tokens, dtype=torch.long),
            "problem_spans": self.problem_spans,
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

    logger.info("Loading real episodic data (multi-task QA)...")

    dataset = RealEpisodicDataset(
        tokenizer=tokenizer,
        num_episodes=getattr(config, "num_episodes_p2", 30000),
        seq_len=config.seq_len,
        num_problems=num_problems,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_episodic,
    )
