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


class DocumentChunkedDataset(IterableDataset):
    """
    Streams text from a HuggingFace dataset, tokenises on the fly,
    and yields **one document per episode** (no cross-document
    concatenation).

    Each document is either:
    - Truncated to ``seq_len`` if longer, or
    - Skipped if shorter than ``min_len`` tokens.

    This preserves within-document coherence which is critical for
    Phase 1 meta-learning: the model needs the adaptation context
    (first 75 %) to be relevant to the evaluation context (last 25 %).

    Parameters
    ----------
    hf_dataset
        A HuggingFace ``IterableDataset`` (``streaming=True``).
    tokenizer
        A HuggingFace tokenizer.
    seq_len : int
        Number of tokens per episode.
    min_len : int
        Minimum document length in tokens.  Shorter docs are skipped.
        Defaults to ``seq_len // 2`` (ensure at least half is real).
    text_column : str
        Name of the text column in the HF dataset.
    pad_token_id : int | None
        Token used for padding short-but-accepted documents.  If None,
        uses ``tokenizer.eos_token_id``.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        seq_len: int = 2048,
        min_len: int | None = None,
        text_column: str = "text",
        pad_token_id: int | None = None,
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_len = min_len if min_len is not None else seq_len // 2
        self.text_column = text_column
        self.pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else getattr(tokenizer, "eos_token_id", 0)
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for example in self.hf_dataset:
            text = example[self.text_column]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            if len(tokens) < self.min_len:
                continue  # skip very short documents

            if len(tokens) >= self.seq_len:
                # Truncate — take a random window for variety
                max_start = len(tokens) - self.seq_len
                start = random.randint(0, max_start) if max_start > 0 else 0
                chunk = tokens[start : start + self.seq_len]
                real_len = self.seq_len
            else:
                # Pad to seq_len (right-pad with pad_token_id)
                real_len = len(tokens)
                chunk = tokens + [self.pad_token_id] * (
                    self.seq_len - real_len
                )

            input_ids = torch.tensor(chunk, dtype=torch.long)

            # Labels: -100 at padding positions so loss ignores them
            labels = input_ids.clone()
            if real_len < self.seq_len:
                labels[real_len:] = -100

            yield {"input_ids": input_ids, "labels": labels}


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

    # Phase 1 meta-learning needs *within-document coherence*:
    # the model adapts on the first 75 % of an episode and is
    # evaluated on the last 25 %.  If those come from different
    # documents, the adaptation signal is pure noise.
    #
    # We therefore:
    #   1. Prioritise long-document corpora (Wikipedia, FineWeb-Edu)
    #      over short-document ones (C4 averages ~500 tokens).
    #   2. Use DocumentChunkedDataset which keeps one document per
    #      episode (no cross-document concatenation).
    PHASE1_SOURCES = [
        {   # Wikipedia: long articles (avg ~2-3K tokens), highly coherent
            "name": "wikimedia/wikipedia",
            "config": "20231101.en",
            "text_column": "text",
        },
        {   # FineWeb-Edu: curated educational content, long, diverse
            "name": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",
            "text_column": "text",
        },
        {   # C4: last resort — shorter docs, cross-doc noise
            "name": getattr(config, "dataset_name", "allenai/c4"),
            "config": getattr(config, "dataset_config", "en"),
            "text_column": getattr(config, "text_column", "text"),
        },
    ]

    hf_ds = None
    text_column = "text"
    for src in PHASE1_SOURCES:
        try:
            logger.info(
                f"Loading streaming dataset: {src['name']} ({src['config']})"
            )
            hf_ds = load_dataset(
                src["name"],
                src["config"],
                split="train",
                streaming=True,
            )
            # Shuffle the stream so consecutive episodes come from
            # diverse topics.  buffer_size=10000 means ~10K documents
            # are held in memory and sampled randomly.  Without this,
            # Wikipedia streams alphabetically by article title.
            hf_ds = hf_ds.shuffle(seed=42, buffer_size=10000)
            text_column = src["text_column"]
            logger.info(f"  → loaded successfully")
            break
        except Exception as e:
            logger.warning(f"  → {src['name']}: failed ({e}), trying next...")

    if hf_ds is None:
        raise RuntimeError(
            "Could not load any Phase 1 text corpus. "
            "Check network connectivity."
        )

    # Use document-aware chunking: each episode = one document.
    # The 75/25 adapt/eval split means eval starts at position
    # int(seq_len * 0.75).  Documents must be long enough that the
    # eval window contains real tokens, otherwise cross-entropy on
    # all-padding (-100) labels produces NaN.
    adapt_len = int(config.seq_len * 0.75)
    chunk_size = getattr(config, "adapt_every_n", 32)
    adapt_len = (adapt_len // chunk_size) * chunk_size
    # Require at least 128 real eval tokens (or 25% of eval window)
    eval_window = config.seq_len - adapt_len
    min_eval_tokens = max(128, eval_window // 4)
    min_doc_len = adapt_len + min_eval_tokens
    logger.info(
        f"DocumentChunkedDataset: seq_len={config.seq_len}, "
        f"adapt_len={adapt_len}, min_doc_len={min_doc_len}"
    )

    chunked = DocumentChunkedDataset(
        hf_dataset=hf_ds,
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        min_len=min_doc_len,
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
            "labels": self.data[idx].clone(),
            "problem_spans": self.problem_spans,
        }


def collate_episodic(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for episodic datasets.

    Stacks ``input_ids`` and ``labels``.  Keeps per-example
    ``problem_spans`` as a list-of-lists since densely-packed episodes
    have different span positions per example.
    """
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "problem_spans": [b["problem_spans"] for b in batch],
    }


# ------------------------------------------------------------------ #
# Real episodic dataset (multi-task QA from HuggingFace)               #
# ------------------------------------------------------------------ #

class RealEpisodicDataset(Dataset):
    """
    Episodic dataset built from real QA datasets with context grouping.

    Loads questions from multiple HuggingFace datasets, formats them as
    ``Question: ...\\nAnswer: ...\\n\\n`` pairs, packs ``num_problems``
    per episode into a single tokenised sequence, and records
    ``problem_spans`` for per-problem loss computation.

    **Context grouping** — sources that support it group QA pairs by
    topic / article / activity so that all problems within an episode
    share related context.  This gives a much cleaner adaptation signal:
    adapting on problems 1-5 about "anatomy" actually helps with eval
    problems 6-8 about "anatomy".

    Grouped sources:
      - ``rajpurkar/squad``     — reading comprehension, grouped by article (~87 K)
      - ``cais/mmlu``           — multi-domain knowledge, grouped by subject (~14 K)
      - ``Rowan/hellaswag``     — commonsense completion, grouped by activity (~40 K)

    Ungrouped sources (same-source sampling):
      - ``openai/gsm8k``        — grade-school math (~7.5 K)
      - ``allenai/ai2_arc``     — science reasoning, Easy + Challenge (~3.4 K)
      - ``tau/commonsense_qa``  — commonsense reasoning (~10 K)
      - ``ybisk/piqa``          — physical intuition (~16 K)
      - ``google/boolq``        — yes / no reading comprehension (~9.4 K)
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
        # ── Multi-domain knowledge (grouped by subject) ──
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
            "grouper": lambda ex: ex.get("subject", ""),
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
        # ── Commonsense completion (grouped by activity) ──
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
            "grouper": lambda ex: ex.get("activity_label", ""),
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
        # ── Reading comprehension (grouped by article) ──
        {
            "name": "rajpurkar/squad",
            "config": None,
            "split": "train",
            "formatter": lambda ex: (
                f"Based on: \"{ex['context'][:400]}\"\n{ex['question']}",
                ex["answers"]["text"][0]
                if ex["answers"].get("text")
                else "",
            ),
            "grouper": lambda ex: ex.get("title", ""),
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

    # Minimum group size — groups smaller than this are merged into
    # an ungrouped fallback bucket for that source.
    MIN_GROUP_SIZE = 4

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

        # Two-level structure: source_groups[i] is a list of context
        # groups for source i.  Each context group is a list of
        # (question, answer) pairs that share related context.
        self.source_groups: list[list[list[tuple[str, str]]]] = []
        self._load_sources()

        if not self.source_groups:
            raise RuntimeError(
                "No QA pairs loaded. Check network connectivity and "
                "dataset availability."
            )

        total_groups = sum(len(sg) for sg in self.source_groups)
        total_pairs = sum(
            sum(len(g) for g in sg) for sg in self.source_groups
        )
        logger.info(
            f"Loaded {total_pairs} QA pairs in {total_groups} context "
            f"groups across {len(self.source_groups)} sources for "
            f"Phase 2 (context-grouped sampling)"
        )

    def _load_sources(self):
        """Load QA pairs into per-source context groups.

        For sources with a ``grouper`` function, QA pairs are split
        into context groups (e.g. by article title, subject, or
        activity).  Groups smaller than ``MIN_GROUP_SIZE`` are merged
        into a fallback group for that source.

        For sources without a ``grouper``, all QA pairs form a single
        group (equivalent to same-source sampling).
        """
        from datasets import load_dataset

        for src in self.SOURCES:
            try:
                logger.info(f"Loading {src['name']} ({src['config']})...")
                ds = load_dataset(
                    src["name"],
                    src["config"],
                    split=src["split"],
                )

                grouper = src.get("grouper")

                if grouper is None:
                    # No grouping — one big group for this source
                    bucket: list[tuple[str, str]] = []
                    for example in ds:
                        try:
                            q, a = src["formatter"](example)
                            if q and a:
                                bucket.append((q.strip(), a.strip()))
                        except (KeyError, IndexError, TypeError):
                            continue
                    if bucket:
                        self.source_groups.append([bucket])
                        logger.info(
                            f"  → {src['name']}: {len(bucket)} pairs "
                            f"(1 group)"
                        )
                    else:
                        logger.warning(
                            f"  → {src['name']}: loaded but 0 valid pairs"
                        )
                else:
                    # Group by key
                    groups_dict: dict[str, list[tuple[str, str]]] = {}
                    for example in ds:
                        try:
                            q, a = src["formatter"](example)
                            key = grouper(example)
                            if q and a and key:
                                groups_dict.setdefault(key, []).append(
                                    (q.strip(), a.strip())
                                )
                        except (KeyError, IndexError, TypeError):
                            continue

                    # Separate large groups from small ones
                    groups: list[list[tuple[str, str]]] = []
                    fallback: list[tuple[str, str]] = []
                    for _key, pairs in groups_dict.items():
                        if len(pairs) >= self.MIN_GROUP_SIZE:
                            groups.append(pairs)
                        else:
                            fallback.extend(pairs)

                    # Merge small groups into a fallback group
                    if len(fallback) >= self.MIN_GROUP_SIZE:
                        groups.append(fallback)

                    if groups:
                        self.source_groups.append(groups)
                        total_pairs = sum(len(g) for g in groups)
                        logger.info(
                            f"  → {src['name']}: {total_pairs} pairs "
                            f"({len(groups)} context groups)"
                        )
                    else:
                        logger.warning(
                            f"  → {src['name']}: loaded but 0 valid "
                            f"groups"
                        )
            except Exception as e:
                logger.warning(
                    f"  → {src['name']}: failed ({e}), skipping"
                )

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Build one episode: num_problems QA pairs packed densely.

        Problems are packed back-to-back with no fixed slot boundaries.
        Each problem is tokenised as::

            Question: {q}\\nAnswer: {a}\\n\\n

        ``problem_spans`` records the actual ``(sol_start, sol_end)``
        positions of each answer in token space, so the loss is
        computed on exactly the answer tokens (no padding waste).

        Remaining space at the end is padded with ``pad_id`` and
        labelled ``-100``.
        """
        rng = random.Random(idx)
        pad_id = self.tokenizer.eos_token_id or 0

        # Two-level context-grouped sampling:
        # 1. Pick a source uniformly (preserves source-level balance)
        # 2. Pick a context group within that source
        # 3. Draw all problems from that group
        # This ensures adapt problems (1-5) share context with eval
        # problems (6-8), so adaptation genuinely helps.
        source_groups = rng.choice(self.source_groups)
        group = rng.choice(source_groups)
        selected = [rng.choice(group) for _ in range(self.num_problems)]

        all_tokens: list[int] = []
        all_labels: list[int] = []
        problem_spans: list[tuple[int, int]] = []

        for q, a in selected:
            q_tokens = self.tokenizer.encode(
                f"Question: {q}\nAnswer: ", add_special_tokens=False,
            )
            a_tokens = self.tokenizer.encode(
                f"{a}\n\n", add_special_tokens=False,
            )

            sol_start = len(all_tokens) + len(q_tokens)
            # Ensure sol_start >= 1 (we need logits at sol_start - 1)
            sol_start = max(sol_start, 1)
            sol_end = sol_start + len(a_tokens)

            # Stop if this problem would overflow seq_len
            if sol_end > self.seq_len:
                # Try to fit a truncated version
                remaining = self.seq_len - len(all_tokens)
                if remaining > len(q_tokens) + 1:
                    # Truncate answer to fit
                    a_budget = remaining - len(q_tokens)
                    a_tokens = a_tokens[:a_budget]
                    sol_end = sol_start + len(a_tokens)
                else:
                    break  # no room for even the question

            all_tokens.extend(q_tokens)
            all_tokens.extend(a_tokens)

            all_labels.extend([-100] * len(q_tokens))
            all_labels.extend(a_tokens)

            problem_spans.append((sol_start, sol_end))

        # Pad to seq_len
        pad_len = self.seq_len - len(all_tokens)
        if pad_len > 0:
            all_tokens.extend([pad_id] * pad_len)
            all_labels.extend([-100] * pad_len)

        # Safety truncate (shouldn't happen but defensive)
        all_tokens = all_tokens[: self.seq_len]
        all_labels = all_labels[: self.seq_len]

        return {
            "input_ids": torch.tensor(all_tokens, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "problem_spans": problem_spans,
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
    p2_batch = getattr(config, "batch_size_p2", config.batch_size)

    if synthetic:
        dataset = SyntheticEpisodicDataset(
            num_episodes=getattr(config, "num_episodes_p2", 1000),
            seq_len=config.seq_len,
            num_problems=num_problems,
            vocab_size=getattr(config, "vocab_size", 1000),
        )
        return DataLoader(
            dataset,
            batch_size=p2_batch,
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
        batch_size=p2_batch,
        num_workers=0,
        collate_fn=collate_episodic,
    )
