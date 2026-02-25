"""
Topic-grouped episode data loading for NAT v2 training.

Each episode is a contiguous window of tokens from one *topic*.
Topics are narrow semantic groups (e.g. "Algebra_Level3",
"algebra__linear_1d") so the adaptive neuron gets coherent material
to adapt to within each episode.

Datasets:
  Primary:   MATH (hendrycks), DeepMind Math
  Secondary: ScienceQA (text-only), DROP

Usage:
    # From HuggingFace (downloads + tokenizes + caches):
    dataset = EpisodeDataset.from_huggingface(tokenizer, cache_dir="data/cache")

    # From pre-tokenized .pt files:
    dataset = EpisodeDataset.from_token_files(["topic_a.pt", "topic_b.pt"])

    # Sample a batch:
    input_ids, topic_indices = dataset.sample_batch(batch_size=4)
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class TopicGroup:
    """One narrow topic with its pre-tokenized text."""
    dataset: str       # e.g. "math", "deepmind_math"
    topic_key: str     # e.g. "Algebra_Level3", "algebra__linear_1d"
    tokens: Tensor     # 1D LongTensor of pre-tokenized text


class EpisodeDataset:
    """
    Manages tokenized topics and samples episodes for training.

    Each topic is a flat 1D tensor of token ids. Episodes are sampled
    as contiguous windows of `seq_len` tokens from a uniformly chosen topic.
    """

    def __init__(
        self,
        topics: List[TopicGroup],
        seq_len: int = 2048,
    ):
        """
        Args:
            topics: list of TopicGroup. Topics with < seq_len tokens are filtered out.
            seq_len: tokens per episode.
        """
        self.seq_len = seq_len
        self.topics = [t for t in topics if len(t.tokens) >= seq_len]

        if not self.topics:
            raise ValueError(
                f"No topics have >= {seq_len} tokens. "
                f"Got {len(topics)} topics, max length "
                f"{max(len(t.tokens) for t in topics) if topics else 0}"
            )

        total_tokens = sum(len(t.tokens) for t in self.topics)
        datasets = {}
        for t in self.topics:
            datasets.setdefault(t.dataset, 0)
            datasets[t.dataset] += 1

        print(
            f"EpisodeDataset: {len(self.topics)} topics "
            f"({len(topics) - len(self.topics)} filtered), "
            f"{total_tokens:,} total tokens"
        )
        for ds_name, count in sorted(datasets.items()):
            print(f"  {ds_name}: {count} topics")

    @classmethod
    def from_huggingface(
        cls,
        tokenizer,
        seq_len: int = 2048,
        cache_dir: Optional[str] = None,
    ) -> "EpisodeDataset":
        """Load all datasets from HuggingFace, group by topic, tokenize, and cache."""
        cache_path = Path(cache_dir) if cache_dir else None
        all_topics: List[TopicGroup] = []

        loaders = [
            ("math", "math_topics.pt", _load_math_topics),
            ("deepmind_math", "deepmind_math_topics.pt", _load_deepmind_math_topics),
            ("scienceqa", "scienceqa_topics.pt", _load_scienceqa_topics),
            ("drop", "drop_topics.pt", _load_drop_topics),
        ]

        for dataset_name, cache_name, loader_fn in loaders:
            # Check cache
            if cache_path:
                cached = cache_path / cache_name
                if cached.exists():
                    topic_dict = torch.load(cached, weights_only=True)
                    topics = [
                        TopicGroup(dataset=dataset_name, topic_key=k, tokens=v)
                        for k, v in topic_dict.items()
                    ]
                    print(
                        f"Loaded cached: {dataset_name} "
                        f"({len(topics)} topics, "
                        f"{sum(len(t.tokens) for t in topics):,} tokens)"
                    )
                    all_topics.extend(topics)
                    continue

            try:
                print(f"Loading {dataset_name}...")
                topic_dict = loader_fn(tokenizer)

                if cache_path:
                    cache_path.mkdir(parents=True, exist_ok=True)
                    torch.save(topic_dict, cached)

                topics = [
                    TopicGroup(dataset=dataset_name, topic_key=k, tokens=v)
                    for k, v in topic_dict.items()
                ]
                print(
                    f"  Loaded: {len(topics)} topics, "
                    f"{sum(len(t.tokens) for t in topics):,} tokens"
                )
                all_topics.extend(topics)

            except Exception as e:
                print(f"  Warning: failed to load {dataset_name}: {e}")
                continue

        if not all_topics:
            raise RuntimeError("No topics loaded. Check dataset availability.")

        return cls(all_topics, seq_len)

    @classmethod
    def from_token_files(
        cls,
        paths: List[str],
        seq_len: int = 2048,
    ) -> "EpisodeDataset":
        """
        Load pre-tokenized .pt files.

        Each file should be either:
          - A dict mapping topic_key → 1D LongTensor
          - A single 1D LongTensor (treated as one topic)
        """
        topics = []
        for path in paths:
            data = torch.load(path, weights_only=True)
            stem = Path(path).stem

            if isinstance(data, dict):
                for key, tokens in data.items():
                    topics.append(
                        TopicGroup(dataset=stem, topic_key=key, tokens=tokens)
                    )
            else:
                topics.append(
                    TopicGroup(dataset=stem, topic_key=stem, tokens=data)
                )

        return cls(topics, seq_len)

    def sample_batch(
        self, batch_size: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Sample a batch of episodes. Each item picks a uniform random topic.

        Returns:
            input_ids: (batch_size, seq_len) LongTensor
            topic_indices: list of topic index per batch item
        """
        episodes = []
        topic_indices = []

        for _ in range(batch_size):
            idx = random.randint(0, len(self.topics) - 1)
            tokens = self.topics[idx].tokens
            start = random.randint(0, len(tokens) - self.seq_len)
            episodes.append(tokens[start : start + self.seq_len])
            topic_indices.append(idx)

        return torch.stack(episodes), topic_indices

    @property
    def num_topics(self) -> int:
        return len(self.topics)


# ── Per-dataset loaders ──────────────────────────────────────────────


def _tokenize_texts(tokenizer, texts: List[str]) -> Tensor:
    """Join texts with double newline and tokenize to a 1D LongTensor."""
    full_text = "\n\n".join(texts)
    return tokenizer.encode(full_text, return_tensors="pt")[0]


def _load_math_topics(tokenizer) -> Dict[str, Tensor]:
    """
    Load MATH benchmark, group by (subject, level).

    Format: Q: {problem}\nA: {solution}\n\n
    """
    from datasets import load_dataset

    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")

    groups: Dict[str, List[str]] = {}
    for row in ds:
        subject = row.get("subject", "unknown")
        level = row.get("level", "unknown")
        topic_key = f"{subject}_Level{level}"
        text = f"Q: {row['problem']}\nA: {row['solution']}"
        groups.setdefault(topic_key, []).append(text)

    result = {}
    for key, texts in groups.items():
        tokens = _tokenize_texts(tokenizer, texts)
        result[key] = tokens
        print(f"    {key}: {len(texts)} problems, {len(tokens):,} tokens")

    return result


def _load_deepmind_math_topics(
    tokenizer, max_per_config: int = 5000,
) -> Dict[str, Tensor]:
    """
    Load DeepMind math_dataset, one topic per config name.

    Format: Q: {question}\nA: {answer}\n\n
    """
    import os
    from datasets import get_dataset_config_names, load_dataset

    # Legacy dataset script requires opt-in; env var works across all versions
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

    configs = get_dataset_config_names("deepmind/math_dataset")
    print(f"    DeepMind Math: {len(configs)} configs")

    result = {}
    for config_name in configs:
        try:
            ds = load_dataset(
                "deepmind/math_dataset", config_name, split="train",
            )
            # Limit examples per config
            if len(ds) > max_per_config:
                ds = ds.select(range(max_per_config))

            texts = [
                f"Q: {row['question']}\nA: {row['answer']}"
                for row in ds
            ]
            tokens = _tokenize_texts(tokenizer, texts)
            result[config_name] = tokens
            print(
                f"    {config_name}: {len(texts)} examples, "
                f"{len(tokens):,} tokens"
            )
        except Exception as e:
            print(f"    Skipping {config_name}: {e}")
            continue

    return result


def _load_scienceqa_topics(
    tokenizer, min_examples: int = 5,
) -> Dict[str, Tensor]:
    """
    Load ScienceQA, filter text-only, group by skill.

    Format: Q: {question}\n{choices}\nA: {lecture}\n{solution}\n\n
    """
    from datasets import load_dataset

    ds = load_dataset("derek-thomas/ScienceQA", split="train")

    groups: Dict[str, List[str]] = {}
    for row in ds:
        # Filter: text-only (no image)
        if row.get("image") is not None:
            continue

        skill = row.get("skill", "unknown")
        if not skill:
            continue

        # Format choices
        choices = row.get("choices", [])
        choices_text = "\n".join(
            f"  ({chr(65 + i)}) {c}" for i, c in enumerate(choices)
        )

        lecture = row.get("lecture", "")
        solution = row.get("solution", "")

        text = f"Q: {row['question']}\n{choices_text}\nA: {lecture}\n{solution}"
        groups.setdefault(skill, []).append(text)

    # Filter skills with too few examples
    result = {}
    for key, texts in groups.items():
        if len(texts) < min_examples:
            continue
        tokens = _tokenize_texts(tokenizer, texts)
        result[key] = tokens

    print(
        f"    ScienceQA: {len(result)} skills "
        f"(filtered {len(groups) - len(result)} with <{min_examples} examples)"
    )
    return result


def _load_drop_topics(tokenizer) -> Dict[str, Tensor]:
    """
    Load DROP, group by section_id. Format passage + all Q&A pairs per section.

    Format: {passage}\nQ: {question}\nA: {spans}\n\n
    """
    from datasets import load_dataset

    ds = load_dataset("ucinlp/drop", split="train")

    # Group all Q&A pairs by section_id
    sections: Dict[str, Dict] = {}
    for row in ds:
        section_id = row.get("section_id", "unknown")
        if section_id not in sections:
            sections[section_id] = {
                "passage": row.get("passage", ""),
                "qas": [],
            }

        # Extract answer spans
        answers_spans = row.get("answers_spans", {})
        if isinstance(answers_spans, dict):
            spans = answers_spans.get("spans", [])
        elif isinstance(answers_spans, list):
            spans = answers_spans
        else:
            spans = []

        if isinstance(spans, list):
            answer_text = ", ".join(str(s) for s in spans)
        else:
            answer_text = str(spans)

        sections[section_id]["qas"].append(
            f"Q: {row.get('question', '')}\nA: {answer_text}"
        )

    result = {}
    for section_id, data in sections.items():
        text = data["passage"] + "\n" + "\n".join(data["qas"])
        tokens = _tokenize_texts(tokenizer, [text])
        result[section_id] = tokens

    print(f"    DROP: {len(result)} sections")
    return result
