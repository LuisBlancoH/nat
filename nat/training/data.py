"""
Data loading and episode construction for NAT training.

Provides:
  - ``build_phase1_dataloader``  — episodic multi-domain data for Phase 1.
  - ``build_text_dataset``       — generic tokenised-chunked dataset.
  - ``EpisodeDataset``           — wraps an iterable HF dataset into
    fixed-length token sequences suitable for episodic training.

Phase 1 data sources (multi-domain with reasoning traces)
---------------------------------------------------------
- camel-ai/math — 50k problems across 25 sub-topics with step-by-step solutions
- Hendrycks MATH (EleutherAI) — competition math across 7 subjects
- MATH-Hard (lighteval) — hard competition math problems
- OpenR1-Math-220k (open-r1) — competition math with CoT, pre-verified correct answers
- CodeForces-CoTs (open-r1) — competitive programming with CoT, grouped by difficulty tier
- DROP — reading comprehension with discrete reasoning
- ScienceQA — science questions grouped by fine-grained skill (379 skills)

Phase 2 uses the same domain datasets for consolidation training.

Synthetic datasets are provided for unit-testing and development.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
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
        self.data: list[torch.Tensor] = []

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
# Multi-domain episodic dataset (Phase 1)                              #
# ------------------------------------------------------------------ #

class MultiDomainEpisodeDataset(Dataset):
    """
    Episodic dataset built from multiple reasoning domains with
    context-grouped sampling.

    Each episode packs ``num_problems`` related problems from the same
    context group (exercise type, algorithm tag, shared passage, etc.)
    into a single tokenised sequence.

    Domains:
      - camel-ai/math      — grouped by sub_topic (~hundreds of exercise types)
      - Hendrycks MATH     — grouped by subject + difficulty level
      - MATH-Hard          — grouped by subject + difficulty level
      - OpenR1-Math-220k   — grouped by problem_type; filtered to verified-correct only
      - CodeForces-CoTs    — grouped by contest_type + index (difficulty tier)
      - DROP               — grouped by shared passage (section_id)
      - ScienceQA          — grouped by skill (379 fine-grained skills)

    Falls back gracefully if a dataset is unavailable.
    """

    SOURCES = [
        # ── Math: camel-ai/math — 50k problems, 25 sub-topics ──
        {
            "name": "camel-ai/math",
            "config": None,
            "domain": "math",
            "split": "train",
            "formatter": lambda ex: (
                ex.get("message_1", ""),
                ex.get("message_2", ""),
            ),
            # sub_topic gives fine-grained groups like "Solving linear equations"
            "grouper": lambda ex: ex.get("sub_topic", ex.get("topic;", "math")),
        },
        # ── Math: Hendrycks MATH (all 7 subjects) ──
        *[
            {
                "name": "EleutherAI/hendrycks_math",
                "config": cfg,
                "domain": "math",
                "split": "train",
                "formatter": lambda ex: (
                    ex.get("problem", ""),
                    ex.get("solution", ""),
                ),
                "grouper": lambda ex: f"{ex.get('type', 'unknown')}_L{ex.get('level', '?')}",
            }
            for cfg in [
                "algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory",
                "prealgebra", "precalculus",
            ]
        ],
        # ── Math (hard): lighteval/MATH-Hard ──
        {
            "name": "lighteval/MATH-Hard",
            "config": None,
            "domain": "math_hard",
            "split": "train",
            "formatter": lambda ex: (
                ex.get("problem", ""),
                ex.get("solution", ""),
            ),
            "grouper": lambda ex: f"{ex.get('type', 'unknown')}_L{ex.get('level', '?')}",
        },
        # ── Code: CodeForces-CoTs ──
        {
            "name": "open-r1/codeforces-cots",
            "config": None,
            "domain": "code",
            "split": "train",
            "formatter": lambda ex: (
                ex.get("description", ex.get("prompt", "")),
                ex.get("generation", ex.get("editorial", "")),
            ),
            # Group by contest_type + problem index (A/B/C/D = difficulty tier).
            # No 'tags' field exists in this dataset; index is the best
            # available difficulty/complexity proxy.
            "grouper": lambda ex: f"{ex.get('contest_type', 'CF')}_{ex.get('index', 'A')}",
        },
        # ── Math (hard): open-r1/OpenR1-Math-220k ──
        # Competition math with full chain-of-thought reasoning.
        # Pre-verified: correctness_count > 0 means at least one
        # generation was confirmed correct by symbolic verifier.
        # Grouped by problem_type (e.g. "algebra", "combinatorics").
        {
            "name": "open-r1/OpenR1-Math-220k",
            "config": None,
            "domain": "math_hard",
            "split": "train",
            "formatter": lambda ex: (
                ex.get("problem", ""),
                ex.get("solution", ""),
            ),
            # Only use examples with at least one verified-correct generation
            "filter": lambda ex: (ex.get("correctness_count") or 0) > 0,
            "grouper": lambda ex: ex.get("problem_type", ex.get("source", "math")),
        },
        # ── Reading: DROP ──
        {
            "name": "ucinlp/drop",
            "config": None,
            "domain": "reading",
            "split": "train",
            "formatter": lambda ex: (
                f"Based on: \"{ex['passage'][:400]}\"\n{ex['question']}",
                ex["answers_spans"]["spans"][0]
                if ex.get("answers_spans")
                and ex["answers_spans"].get("spans")
                else "",
            ),
            "grouper": lambda ex: ex.get("section_id", ex.get("passage", "")[:200]),
        },
        # ── Science: ScienceQA ──
        {
            "name": "derek-thomas/ScienceQA",
            "config": None,
            "domain": "science",
            "split": "train",
            "formatter": lambda ex: (
                (
                    (f"Context: {ex['hint']}\n" if ex.get("hint") else "")
                    + ex.get("question", "")
                ),
                (
                    ex["choices"][ex["answer"]]
                    if isinstance(ex.get("answer"), int)
                    and 0 <= ex["answer"] < len(ex.get("choices", []))
                    else ""
                ),
            ),
            # skill has 379 fine-grained groups e.g. "Identify the life cycle of a butterfly"
            "grouper": lambda ex: ex.get("skill", ex.get("category", "science")),
        },
    ]

    # Also try local datasets with pre-generated CoT traces
    LOCAL_SOURCES = [
        {
            "name": "AR-LSAT (local, CoT-enriched)",
            "local_path": "data/ar_lsat_cot.json",
            "domain": "logic",
            "formatter": lambda ex: (
                ex.get("problem", ex.get("question", "")),
                ex.get("solution", ex.get("answer", "")),
            ),
            "grouper": lambda ex: ex.get("scenario_id", ex.get("problem", "")[:200]),
        },
        {
            "name": "DROP (local, CoT-enriched)",
            "local_path": "data/drop_cot.json",
            "domain": "reading",
            "formatter": lambda ex: (
                f"Based on: \"{ex.get('passage', '')[:400]}\"\n{ex.get('question', '')}",
                ex.get("solution", ex.get("answer", "")),
            ),
            "grouper": lambda ex: ex.get("passage", "")[:200],
        },
    ]

    # Minimum group size — groups smaller than this are merged
    MIN_GROUP_SIZE = 4

    def __init__(
        self,
        tokenizer,
        num_episodes: int = 50000,
        seq_len: int = 2048,
        num_problems: int = 8,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_episodes = num_episodes
        self.seq_len = seq_len
        self.num_problems = num_problems

        # domain_groups[domain_name] = list of context groups
        # Each context group is a list of (problem, solution) pairs
        self.domain_groups: dict[str, list[list[tuple[str, str]]]] = {}
        self._load_sources()

        if not self.domain_groups:
            raise RuntimeError(
                "No QA pairs loaded. Check network connectivity and "
                "dataset availability."
            )

        total_groups = sum(len(gs) for gs in self.domain_groups.values())
        total_pairs = sum(
            sum(len(g) for g in gs) for gs in self.domain_groups.values()
        )
        logger.info(
            f"Loaded {total_pairs} problem-solution pairs in "
            f"{total_groups} context groups across "
            f"{len(self.domain_groups)} domains for Phase 1"
        )
        for domain, groups in self.domain_groups.items():
            n_pairs = sum(len(g) for g in groups)
            logger.info(
                f"  {domain}: {n_pairs} pairs ({len(groups)} groups)"
            )

    def _load_sources(self):
        """Load QA pairs from all available sources."""
        from datasets import load_dataset

        loaded: list[str] = []
        failed: list[str] = []

        for src in self.SOURCES:
            try:
                logger.info(f"Loading {src['name']}...")
                load_kwargs: dict[str, Any] = {}
                if src.get("config"):
                    load_kwargs["name"] = src["config"]
                if src.get("trust_remote_code"):
                    load_kwargs["trust_remote_code"] = True

                ds = load_dataset(
                    src["name"],
                    split=src["split"],
                    **load_kwargs,
                )

                domain = src["domain"]
                formatter = src["formatter"]
                grouper = src.get("grouper")
                filt = src.get("filter")  # optional predicate

                groups_dict: dict[str, list[tuple[str, str]]] = {}
                for example in ds:
                    try:
                        if filt is not None and not filt(example):
                            continue
                        q, a = formatter(example)
                        if not q or not a:
                            continue

                        key = str(grouper(example)) if grouper else "_all"
                        if not key:
                            continue

                        groups_dict.setdefault(key, []).append(
                            (q.strip(), a.strip())
                        )
                    except (KeyError, IndexError, TypeError, ValueError):
                        continue

                # Separate large and small groups
                groups: list[list[tuple[str, str]]] = []
                fallback: list[tuple[str, str]] = []
                for _key, pairs in groups_dict.items():
                    if len(pairs) >= self.MIN_GROUP_SIZE:
                        groups.append(pairs)
                    else:
                        fallback.extend(pairs)

                if len(fallback) >= self.MIN_GROUP_SIZE:
                    groups.append(fallback)

                if groups:
                    if domain not in self.domain_groups:
                        self.domain_groups[domain] = []
                    self.domain_groups[domain].extend(groups)
                    total = sum(len(g) for g in groups)
                    logger.info(
                        f"  → {src['name']}: {total} pairs "
                        f"({len(groups)} groups) [domain={domain}]"
                    )
                    loaded.append(src["name"])
                else:
                    logger.warning(
                        f"  → {src['name']}: loaded but 0 valid groups"
                    )
                    failed.append(src["name"])
            except Exception as e:
                logger.warning(f"  → {src['name']}: failed ({e}), skipping")
                failed.append(src["name"])

        # Try local CoT-enriched datasets
        for src in self.LOCAL_SOURCES:
            try:
                local_path = src["local_path"]
                if not os.path.isabs(local_path):
                    project_root = os.path.dirname(
                        os.path.dirname(os.path.dirname(__file__))
                    )
                    local_path = os.path.join(project_root, local_path)

                if not os.path.exists(local_path):
                    logger.info(
                        f"  → {src['name']}: not found at {local_path} (optional)"
                    )
                    continue

                logger.info(f"Loading local: {src['name']}...")
                with open(local_path, "r") as f:
                    data = json.load(f)

                domain = src["domain"]
                formatter = src["formatter"]
                grouper = src.get("grouper")

                groups_dict: dict[str, list[tuple[str, str]]] = {}
                for example in data:
                    try:
                        q, a = formatter(example)
                        if not q or not a:
                            continue
                        key = str(grouper(example)) if grouper else "_all"
                        groups_dict.setdefault(key, []).append(
                            (q.strip(), a.strip())
                        )
                    except (KeyError, IndexError, TypeError, ValueError):
                        continue

                groups = []
                fallback = []
                for _key, pairs in groups_dict.items():
                    if len(pairs) >= self.MIN_GROUP_SIZE:
                        groups.append(pairs)
                    else:
                        fallback.extend(pairs)
                if len(fallback) >= self.MIN_GROUP_SIZE:
                    groups.append(fallback)

                if groups:
                    if domain not in self.domain_groups:
                        self.domain_groups[domain] = []
                    self.domain_groups[domain].extend(groups)
                    total = sum(len(g) for g in groups)
                    logger.info(
                        f"  → {src['name']}: {total} pairs "
                        f"({len(groups)} groups) [domain={domain}]"
                    )
                    loaded.append(src["name"])
            except Exception as e:
                logger.warning(f"  → {src['name']}: failed ({e}), skipping")

        # Summary
        n_total = len(self.SOURCES) + len(self.LOCAL_SOURCES)
        n_ok = len(loaded)
        n_fail = len(failed)
        logger.info("")
        logger.info(f"Phase 1 dataset summary: {n_ok}/{n_total} loaded, {n_fail} failed")
        if loaded:
            logger.info(f"  ✓ Loaded: {', '.join(loaded)}")
        if failed:
            logger.warning(f"  ✗ Failed: {', '.join(failed)}")
        logger.info("")

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Build one episode: num_problems related problems packed densely.

        Two-level sampling:
        1. Pick a domain uniformly (balances domain representation)
        2. Pick a context group within that domain
        3. Draw all problems from that group

        Format: plain text, problem + step-by-step solution.
        No chat template, no <think> tags.
        """
        rng = random.Random(idx)
        pad_id = self.tokenizer.eos_token_id or 0

        # Pick a domain uniformly
        domain = rng.choice(list(self.domain_groups.keys()))
        domain_grps = self.domain_groups[domain]
        group = rng.choice(domain_grps)
        selected = [rng.choice(group) for _ in range(self.num_problems)]

        all_tokens: list[int] = []
        all_labels: list[int] = []
        problem_spans: list[tuple[int, int]] = []

        for q, a in selected:
            q_tokens = self.tokenizer.encode(
                f"Problem: {q}\nSolution: ", add_special_tokens=False,
            )
            a_tokens = self.tokenizer.encode(
                f"{a}\n\n", add_special_tokens=False,
            )

            sol_start = len(all_tokens) + len(q_tokens)
            sol_start = max(sol_start, 1)
            sol_end = sol_start + len(a_tokens)

            if sol_end > self.seq_len:
                remaining = self.seq_len - len(all_tokens)
                if remaining > len(q_tokens) + 1:
                    a_budget = remaining - len(q_tokens)
                    a_tokens = a_tokens[:a_budget]
                    sol_end = sol_start + len(a_tokens)
                else:
                    break

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

        all_tokens = all_tokens[: self.seq_len]
        all_labels = all_labels[: self.seq_len]

        return {
            "input_ids": torch.tensor(all_tokens, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "problem_spans": problem_spans,
            "domain": domain,
        }


# ------------------------------------------------------------------ #
# Domain text dataset (Phase 2 consolidation)                          #
# ------------------------------------------------------------------ #

class SyntheticDomainDataset(Dataset):
    """
    Domain-specific synthetic data for Phase 2 consolidation training.

    Each domain produces token sequences with a characteristic
    statistical profile.
    """

    DOMAINS = [
        "math", "math_hard", "code", "reading", "science",
    ]

    def __init__(
        self,
        domain: str,
        num_episodes: int = 64,
        seq_len: int = 256,
        vocab_size: int = 1000,
        seed: int = 42,
    ):
        super().__init__()
        self.domain = domain
        self.num_episodes = num_episodes
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        if domain in self.DOMAINS:
            domain_offset = self.DOMAINS.index(domain)
        else:
            domain_offset = hash(domain) % 100

        rng = torch.Generator().manual_seed(seed + domain_offset * 1000)
        num_domains = max(len(self.DOMAINS), domain_offset + 1)
        slice_size = vocab_size // num_domains
        domain_start = domain_offset * slice_size
        domain_end = min(domain_start + slice_size, vocab_size)
        domain_end = max(domain_end, domain_start + 10)
        domain_end = min(domain_end, vocab_size)

        self.data: list[torch.Tensor] = []
        num_domain_tokens = int(seq_len * 0.7)
        num_random_tokens = seq_len - num_domain_tokens

        for _ in range(num_episodes):
            domain_tokens = torch.randint(
                domain_start, domain_end,
                (num_domain_tokens,), generator=rng,
            )
            random_tokens = torch.randint(
                0, vocab_size,
                (num_random_tokens,), generator=rng,
            )
            tokens = torch.cat([domain_tokens, random_tokens])
            perm = torch.randperm(seq_len, generator=rng)
            tokens = tokens[perm]
            self.data.append(tokens)

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.data[idx]}


# ------------------------------------------------------------------ #
# Domain names                                                         #
# ------------------------------------------------------------------ #

DOMAINS: list[str] = [
    "math", "math_hard", "code", "reading", "science",
]
"""Default domain names for Phase 2 consolidation training."""


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
    Build a ``DataLoader`` for Phase 1 episodic training.

    Parameters
    ----------
    config : NATConfig
        Must have ``seq_len``, ``batch_size``, ``num_episodes_p1``,
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
            num_episodes=getattr(config, "num_episodes_p1", 1000),
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

    # ---- Real multi-domain episodic data ----
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

    logger.info("Loading real episodic data (multi-domain)...")

    dataset = MultiDomainEpisodeDataset(
        tokenizer=tokenizer,
        num_episodes=getattr(config, "num_episodes_p1", 50000),
        seq_len=config.seq_len,
        num_problems=num_problems,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=0,
        collate_fn=collate_episodic,
    )


def build_domain_dataloader(
    config,
    domain: str,
    tokenizer=None,
    *,
    synthetic: bool = True,
) -> DataLoader | None:
    """
    Build a ``DataLoader`` for a single domain (Phase 2 consolidation).

    Parameters
    ----------
    config : NATConfig
    domain : str
    tokenizer : optional
    synthetic : bool
        If ``True``, use ``SyntheticDomainDataset``.

    Returns
    -------
    DataLoader or None
    """
    if synthetic:
        sessions_per_domain = getattr(config, "sessions_per_domain_p2", 20)
        dataset = SyntheticDomainDataset(
            domain=domain,
            num_episodes=max(sessions_per_domain * 4, 64),
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

    # Real domain data — load from same sources as Phase 1
    assert tokenizer is not None, (
        "tokenizer is required for non-synthetic data."
    )

    from datasets import load_dataset

    # Map domain names to source datasets
    DOMAIN_SOURCE_MAP: dict[str, list[dict]] = {
        "math": [
            {
                "name": "camel-ai/math",
                "config": None,
                "split": "train",
                "formatter": lambda ex: (
                    f"Problem: {ex.get('message_1', '')}\n"
                    f"Solution: {ex.get('message_2', '')}\n\n"
                ),
            },
            {
                "name": "EleutherAI/hendrycks_math",
                "config": "algebra",
                "split": "train",
                "formatter": lambda ex: (
                    f"Problem: {ex.get('problem', '')}\n"
                    f"Solution: {ex.get('solution', '')}\n\n"
                ),
            },
        ],
        "math_hard": [
            {
                "name": "lighteval/MATH-Hard",
                "config": None,
                "split": "train",
                "formatter": lambda ex: (
                    f"Problem: {ex.get('problem', '')}\n"
                    f"Solution: {ex.get('solution', '')}\n\n"
                ),
            },
            {
                "name": "open-r1/OpenR1-Math-220k",
                "config": None,
                "split": "train",
                "formatter": lambda ex: (
                    f"Problem: {ex.get('problem', '')}\n"
                    f"Solution: {ex.get('solution', '')}\n\n"
                ),
                "filter": lambda ex: (ex.get("correctness_count") or 0) > 0,
            },
        ],
        "code": [
            {
                "name": "open-r1/codeforces-cots",
                "config": None,
                "split": "train",
                "formatter": lambda ex: (
                    f"Problem: {ex.get('description', ex.get('prompt', ''))}\n"
                    f"Solution: {ex.get('generation', ex.get('editorial', ''))}\n\n"
                ),
            },
        ],
        "reading": [
            {
                "name": "ucinlp/drop",
                "config": None,
                "split": "train",
                "formatter": lambda ex: (
                    f"Passage: {ex['passage'][:400]}\n"
                    f"Question: {ex['question']}\n"
                    f"Answer: {ex['answers_spans']['spans'][0] if ex.get('answers_spans') and ex['answers_spans'].get('spans') else ''}\n\n"
                ),
            },
        ],
        "science": [
            {
                "name": "derek-thomas/ScienceQA",
                "config": None,
                "split": "train",
                "formatter": lambda ex: (
                    f"Question: {ex.get('question', '')}\n"
                    f"Answer: {ex['choices'][ex['answer']] if isinstance(ex.get('answer'), int) and 0 <= ex['answer'] < len(ex.get('choices', [])) else ''}\n\n"
                ),
            },
        ],
    }

    if domain not in DOMAIN_SOURCE_MAP:
        logger.warning(f"Unknown domain '{domain}' for real data")
        return None

    sources = DOMAIN_SOURCE_MAP[domain]
    sessions_per_domain = getattr(config, "sessions_per_domain_p2", 20)
    num_episodes = max(sessions_per_domain * 4, 64)

    # Load from HuggingFace and tokenize into chunks
    from nat.training.phase2_consolidation import DomainTextDataset

    for src in sources:
        try:
            logger.info(f"Loading domain '{domain}' source: {src['name']}...")
            load_kwargs: dict[str, Any] = {"split": src["split"]}
            if src["config"]:
                load_kwargs["name"] = src["config"]

            hf_ds = load_dataset(src["name"], **load_kwargs)

            domain_dataset = DomainTextDataset(
                hf_dataset=hf_ds,
                tokenizer=tokenizer,
                formatter=src["formatter"],
                seq_len=config.seq_len,
                num_episodes=num_episodes,
                streaming=False,
            )

            return DataLoader(
                domain_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )
        except Exception as e:
            logger.warning(f"  → {src['name']}: failed ({e}), skipping")

    logger.warning(f"No sources loaded for domain '{domain}'")
    return None


def build_domain_sequence(
    config,
    domains: list[str] | None = None,
) -> list[str]:
    """
    Build a domain sequence for one Phase 2 consolidation run.

    Default structure::

        D1 × N  →  D2 × N  →  D1 × K

    Parameters
    ----------
    config : NATConfig
    domains : list[str], optional

    Returns
    -------
    list[str]
    """
    if domains is None:
        domains = DOMAINS

    sessions_per_domain = getattr(config, "sessions_per_domain_p2", 20)
    forgetting_sessions = getattr(config, "forgetting_test_sessions_p2", 5)

    d1, d2 = random.sample(domains, 2)

    sequence: list[str] = (
        [d1] * sessions_per_domain
        + [d2] * sessions_per_domain
        + [d1] * forgetting_sessions
    )
    return sequence


def collate_episodes(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Default collate that stacks ``input_ids`` tensors."""
    return {"input_ids": torch.stack([b["input_ids"] for b in batch])}
