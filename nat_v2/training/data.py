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
            ("gsm8k", "gsm8k_topics.pt", _load_gsm8k_topics),
            ("arc", "arc_topics.pt", _load_arc_topics),
            ("openbookqa", "openbookqa_topics.pt", _load_openbookqa_topics),
            ("mmlu", "mmlu_topics.pt", _load_mmlu_topics),
            ("aqua_rat", "aqua_rat_topics.pt", _load_aqua_rat_topics),
            # logiqa skipped: uses legacy dataset script incompatible with datasets 4.0+
            ("commonsenseqa", "commonsenseqa_topics.pt", _load_commonsenseqa_topics),
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

    def split_topics(
        self, held_out_fraction: float = 0.13, seed: int = 42,
    ) -> "EpisodeDataset":
        """
        Hold out a fraction of topics for verification.

        Removes held-out topics from self and returns a new EpisodeDataset
        containing only those topics. Split is deterministic (seeded).

        Args:
            held_out_fraction: fraction of topics to hold out.
            seed: random seed for reproducible split.

        Returns:
            verify_dataset: EpisodeDataset with held-out topics only.
        """
        rng = random.Random(seed)
        indices = list(range(len(self.topics)))
        rng.shuffle(indices)

        n_held_out = max(1, int(len(indices) * held_out_fraction))
        held_out_idx = set(indices[:n_held_out])

        train_topics = [t for i, t in enumerate(self.topics) if i not in held_out_idx]
        verify_topics = [t for i, t in enumerate(self.topics) if i in held_out_idx]

        # Update self in-place to only contain train topics
        self.topics = train_topics
        print(
            f"Topic split: {len(train_topics)} train, "
            f"{len(verify_topics)} verify (held-out)"
        )

        # Return new dataset with held-out topics (skip filtering, already filtered)
        verify_ds = EpisodeDataset.__new__(EpisodeDataset)
        verify_ds.seq_len = self.seq_len
        verify_ds.topics = verify_topics
        return verify_ds

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
    tokenizer,
    max_per_split: int = 50_000,
    chunk_size: int = 2000,
) -> Dict[str, Tensor]:
    """
    Load DeepMind math via parquet re-upload, grouped into sub-topics.

    The original deepmind/math_dataset uses a legacy loading script
    that no longer works. This loads from a parquet re-upload and
    chunks each difficulty split into sub-topics.

    Format: Q: {question}\nA: {answer}\n\n
    """
    from datasets import load_dataset

    print(f"    DeepMind Math (parquet): loading train split")

    result = {}
    try:
        ds = load_dataset(
            "midwestern-simulation/that-one-google-math-dataset",
            split="train",
        )
        if len(ds) > max_per_split * 3:
            ds = ds.shuffle(seed=42).select(range(max_per_split * 3))

        texts = [f"Q: {row['q']}\nA: {row['a']}" for row in ds]

        # Chunk into sub-topics for diversity
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            if len(chunk) < chunk_size // 2:
                break
            topic_key = f"dm_math_{i // chunk_size:03d}"
            tokens = _tokenize_texts(tokenizer, chunk)
            result[topic_key] = tokens

        print(
            f"    DeepMind Math: {len(texts)} examples → "
            f"{len(result)} sub-topics"
        )
    except Exception as e:
        print(f"    Failed to load DeepMind Math: {e}")

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


def _load_gsm8k_topics(
    tokenizer, chunk_size: int = 500,
) -> Dict[str, Tensor]:
    """
    Load GSM8K (grade school math word problems), chunked into sub-topics.

    Format: Q: {question}\nA: {answer}\n\n
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")

    texts = [f"Q: {row['question']}\nA: {row['answer']}" for row in ds]

    result = {}
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        if len(chunk) < chunk_size // 2:
            break
        topic_key = f"gsm8k_{i // chunk_size:03d}"
        tokens = _tokenize_texts(tokenizer, chunk)
        result[topic_key] = tokens

    print(f"    GSM8K: {len(texts)} problems → {len(result)} sub-topics")
    return result


def _load_arc_topics(tokenizer) -> Dict[str, Tensor]:
    """
    Load ARC (AI2 Reasoning Challenge), one topic per config+split.

    Format: Q: {question}\n{choices}\nA: {answer}\n\n
    """
    from datasets import load_dataset

    result = {}
    for config in ["ARC-Challenge", "ARC-Easy"]:
        try:
            ds = load_dataset("allenai/ai2_arc", config, split="train")

            texts = []
            for row in ds:
                choices = row.get("choices", {})
                labels = choices.get("label", [])
                choice_texts = choices.get("text", [])
                choices_fmt = "\n".join(
                    f"  ({l}) {t}" for l, t in zip(labels, choice_texts)
                )
                answer_key = row.get("answerKey", "")
                # Find answer text
                answer_text = answer_key
                if answer_key in labels:
                    idx = labels.index(answer_key)
                    if idx < len(choice_texts):
                        answer_text = f"{answer_key}: {choice_texts[idx]}"

                texts.append(
                    f"Q: {row['question']}\n{choices_fmt}\nA: {answer_text}"
                )

            tokens = _tokenize_texts(tokenizer, texts)
            result[config] = tokens
            print(f"    {config}: {len(texts)} questions, {len(tokens):,} tokens")
        except Exception as e:
            print(f"    Skipping {config}: {e}")

    return result


def _load_openbookqa_topics(tokenizer) -> Dict[str, Tensor]:
    """
    Load OpenBookQA with fact1 context.

    Format: Fact: {fact1}\nQ: {question}\n{choices}\nA: {answer}\n\n
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/openbookqa", "additional", split="train")

    texts = []
    for row in ds:
        choices = row.get("choices", {})
        labels = choices.get("label", [])
        choice_texts = choices.get("text", [])
        choices_fmt = "\n".join(
            f"  ({l}) {t}" for l, t in zip(labels, choice_texts)
        )
        answer_key = row.get("answerKey", "")
        answer_text = answer_key
        if answer_key in labels:
            idx = labels.index(answer_key)
            if idx < len(choice_texts):
                answer_text = f"{answer_key}: {choice_texts[idx]}"

        fact = row.get("fact1", "")
        texts.append(
            f"Fact: {fact}\nQ: {row['question_stem']}\n{choices_fmt}\nA: {answer_text}"
        )

    tokens = _tokenize_texts(tokenizer, texts)
    # Single topic — all OpenBookQA together
    result = {"openbookqa_all": tokens}
    print(f"    OpenBookQA: {len(texts)} questions, {len(tokens):,} tokens")
    return result


def _load_mmlu_topics(tokenizer) -> Dict[str, Tensor]:
    """
    Load MMLU, one topic per subject (57 subjects).

    Format: Q: {question}\n{choices}\nA: {answer}\n\n
    """
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")

    groups: Dict[str, List[str]] = {}
    for row in ds:
        subject = row.get("subject", "unknown")
        choices = row.get("choices", [])
        choices_fmt = "\n".join(
            f"  ({chr(65 + i)}) {c}" for i, c in enumerate(choices)
        )
        answer_idx = row.get("answer", 0)
        answer_letter = chr(65 + answer_idx)
        answer_text = answer_letter
        if answer_idx < len(choices):
            answer_text = f"{answer_letter}: {choices[answer_idx]}"

        text = f"Q: {row['question']}\n{choices_fmt}\nA: {answer_text}"
        groups.setdefault(subject, []).append(text)

    result = {}
    for subject, texts in groups.items():
        tokens = _tokenize_texts(tokenizer, texts)
        result[subject] = tokens

    print(
        f"    MMLU: {len(result)} subjects, "
        f"{sum(len(t) for t in result.values()):,} tokens"
    )
    return result


def _load_aqua_rat_topics(
    tokenizer, chunk_size: int = 5000,
) -> Dict[str, Tensor]:
    """
    Load AQuA-RAT (algebraic word problems with rationales), chunked into sub-topics.

    Format: Q: {question}\n{options}\nA: {correct}\nRationale: {rationale}\n\n
    """
    from datasets import load_dataset

    ds = load_dataset("deepmind/aqua_rat", "raw", split="train")

    texts = []
    for row in ds:
        options = row.get("options", [])
        options_fmt = "\n".join(f"  {o}" for o in options)
        texts.append(
            f"Q: {row['question']}\n{options_fmt}\n"
            f"A: {row['correct']}\nRationale: {row.get('rationale', '')}"
        )

    result = {}
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        if len(chunk) < chunk_size // 2:
            break
        topic_key = f"aqua_rat_{i // chunk_size:03d}"
        tokens = _tokenize_texts(tokenizer, chunk)
        result[topic_key] = tokens

    print(f"    AQuA-RAT: {len(texts)} problems → {len(result)} sub-topics")
    return result


def _load_logiqa_topics(tokenizer) -> Dict[str, Tensor]:
    """
    Load LogiQA (logical reasoning).

    Format: Context: {context}\nQ: {query}\n{options}\nA: {correct}\n\n
    """
    from datasets import load_dataset

    ds = load_dataset("lucasmccabe/logiqa", split="train")

    texts = []
    for row in ds:
        options = row.get("options", [])
        options_fmt = "\n".join(
            f"  ({chr(65 + i)}) {o}" for i, o in enumerate(options)
        )
        correct_idx = row.get("correct_option", 0)
        answer_letter = chr(65 + correct_idx)
        answer_text = answer_letter
        if correct_idx < len(options):
            answer_text = f"{answer_letter}: {options[correct_idx]}"

        texts.append(
            f"Context: {row.get('context', '')}\n"
            f"Q: {row.get('query', '')}\n{options_fmt}\nA: {answer_text}"
        )

    tokens = _tokenize_texts(tokenizer, texts)
    result = {"logiqa_all": tokens}
    print(f"    LogiQA: {len(texts)} questions, {len(tokens):,} tokens")
    return result


def _load_commonsenseqa_topics(
    tokenizer, min_examples: int = 5,
) -> Dict[str, Tensor]:
    """
    Load CommonsenseQA, grouped by question_concept.

    Format: Q: {question}\n{choices}\nA: {answer}\n\n
    """
    from datasets import load_dataset

    ds = load_dataset("tau/commonsense_qa", split="train")

    groups: Dict[str, List[str]] = {}
    for row in ds:
        concept = row.get("question_concept", "unknown")
        choices = row.get("choices", {})
        labels = choices.get("label", [])
        choice_texts = choices.get("text", [])
        choices_fmt = "\n".join(
            f"  ({l}) {t}" for l, t in zip(labels, choice_texts)
        )
        answer_key = row.get("answerKey", "")
        answer_text = answer_key
        if answer_key in labels:
            idx = labels.index(answer_key)
            if idx < len(choice_texts):
                answer_text = f"{answer_key}: {choice_texts[idx]}"

        text = f"Q: {row['question']}\n{choices_fmt}\nA: {answer_text}"
        groups.setdefault(concept, []).append(text)

    # Filter concepts with too few examples, merge small ones
    result = {}
    small_texts: List[str] = []
    for concept, texts in groups.items():
        if len(texts) < min_examples:
            small_texts.extend(texts)
        else:
            tokens = _tokenize_texts(tokenizer, texts)
            result[concept] = tokens

    # Merge small concepts into one topic
    if small_texts:
        tokens = _tokenize_texts(tokenizer, small_texts)
        result["misc_concepts"] = tokens

    print(
        f"    CommonsenseQA: {len(groups)} concepts → {len(result)} topics "
        f"({len(small_texts)} examples merged into misc)"
    )
    return result
