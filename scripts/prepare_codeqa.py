"""Download and prepare CodeQA dataset for NAT Phase 2 training.

CodeQA (Liu & Wan, Findings of EMNLP 2021) contains 189,863 QA pairs
about code snippets — 119,778 Java + 70,085 Python.  Multiple questions
share the same code snippet, making it suitable for context-grouped
episodic training.

Usage
-----
Option A — automatic download (requires ``gdown``):

    pip install gdown
    python scripts/prepare_codeqa.py --download --output data/codeqa

Option B — manual download:

    1. Go to the Google Drive link:
       https://drive.google.com/drive/folders/1i04sJNUHwMuDfMV2UfWeQG-Uv8MRw_qh
    2. Download and extract to a directory (e.g. ``data/codeqa_raw/``).
       Expected structure after extraction::

           data/codeqa_raw/
             java/
               train/
                 train.code
                 train.question
                 train.answer
             python/
               train/
                 train.code
                 train.question
                 train.answer

    3. Run:
       python scripts/prepare_codeqa.py --input data/codeqa_raw --output data/codeqa

Optionally push to HuggingFace Hub for easy access from remote machines::

    python scripts/prepare_codeqa.py --input data/codeqa_raw \\
        --output data/codeqa --push-to-hub YOUR_USERNAME/codeqa
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

GDRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/"
    "1i04sJNUHwMuDfMV2UfWeQG-Uv8MRw_qh"
)


# ------------------------------------------------------------------ #
# Download helpers                                                     #
# ------------------------------------------------------------------ #

def download_from_gdrive(output_dir: str) -> str:
    """Download CodeQA from Google Drive using ``gdown``."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        logger.error(
            "gdown is not installed.  Install it with:\n"
            "    pip install gdown\n"
            "Or download manually — see --help."
        )
        sys.exit(1)

    raw_dir = os.path.join(output_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    logger.info("Downloading CodeQA from Google Drive …")
    logger.info("  (this may take a few minutes)")
    gdown.download_folder(GDRIVE_FOLDER, output=raw_dir, quiet=False)

    # gdown may create a subfolder — walk the tree to find the
    # java/train/ directory and return its grandparent.
    for root, dirs, _files in os.walk(raw_dir):
        if "java" in dirs:
            java_train = os.path.join(root, "java", "train")
            if os.path.isdir(java_train):
                logger.info(f"  → found data root at {root}")
                return root

    # Fallback: return raw_dir itself
    return raw_dir


# ------------------------------------------------------------------ #
# Parsing                                                              #
# ------------------------------------------------------------------ #

def read_parallel_files(
    base_dir: str,
    lang: str,
    split: str = "train",
) -> list[dict[str, str]]:
    """Read line-aligned code / question / answer files."""
    split_dir = os.path.join(base_dir, lang, split)

    code_file = os.path.join(split_dir, f"{split}.code")
    question_file = os.path.join(split_dir, f"{split}.question")
    answer_file = os.path.join(split_dir, f"{split}.answer")

    for f in [code_file, question_file, answer_file]:
        if not os.path.exists(f):
            logger.warning(f"  File not found: {f}")
            return []

    with (
        open(code_file, encoding="utf-8") as cf,
        open(question_file, encoding="utf-8") as qf,
        open(answer_file, encoding="utf-8") as af,
    ):
        codes = [line.strip() for line in cf]
        questions = [line.strip() for line in qf]
        answers = [line.strip() for line in af]

    if not (len(codes) == len(questions) == len(answers)):
        logger.error(
            f"Mismatched line counts for {lang}/{split}: "
            f"code={len(codes)}, question={len(questions)}, "
            f"answer={len(answers)}"
        )
        return []

    examples: list[dict[str, str]] = []
    for code, question, answer in zip(codes, questions, answers):
        if code and question and answer:
            examples.append(
                {
                    "code": code,
                    "question": question,
                    "answer": answer,
                    "language": lang,
                }
            )

    logger.info(f"  {lang}/{split}: {len(examples)} QA pairs")
    return examples


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CodeQA dataset for NAT Phase 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Directory containing extracted CodeQA data "
        "(with java/ and python/ subdirectories).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/codeqa",
        help="Output directory for the HuggingFace Dataset "
        "(default: data/codeqa).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Automatically download from Google Drive using gdown.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Push processed dataset to HuggingFace Hub "
        "(e.g. 'your-username/codeqa').",
    )
    args = parser.parse_args()

    if args.download:
        raw_dir = download_from_gdrive(args.output)
    elif args.input:
        raw_dir = args.input
    else:
        parser.error("Provide either --input DIR or --download.")

    # ── Read all train data ──
    logger.info("Reading parallel text files …")
    all_examples: list[dict[str, str]] = []
    for lang in ["java", "python"]:
        examples = read_parallel_files(raw_dir, lang, "train")
        all_examples.extend(examples)

    if not all_examples:
        logger.error(
            "No examples loaded!  Check the input directory structure.\n"
            "Expected:  <input>/java/train/train.code  (etc.)"
        )
        sys.exit(1)

    logger.info(f"\nTotal QA pairs: {len(all_examples):,}")

    # ── Grouping statistics ──
    code_counts = Counter(ex["code"] for ex in all_examples)
    logger.info(f"Unique code snippets: {len(code_counts):,}")
    logger.info(
        f"Average Qs / snippet: "
        f"{len(all_examples) / len(code_counts):.2f}"
    )

    for threshold in [2, 3, 4, 5, 8]:
        rich = {k: v for k, v in code_counts.items() if v >= threshold}
        rich_qs = sum(rich.values())
        logger.info(
            f"  ≥{threshold} Qs: {len(rich):>6,} groups "
            f"({rich_qs:>7,} QA pairs)"
        )

    # ── Save as HuggingFace Dataset ──
    try:
        from datasets import Dataset
    except ImportError:
        logger.error("datasets not installed.  pip install datasets")
        sys.exit(1)

    ds = Dataset.from_list(all_examples)
    os.makedirs(args.output, exist_ok=True)
    ds.save_to_disk(args.output)
    logger.info(f"\nSaved HuggingFace Dataset to {args.output}/")
    logger.info(f"  Columns: {ds.column_names}")
    logger.info(f"  Rows:    {len(ds):,}")

    if args.push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub: {args.push_to_hub} …")
        ds.push_to_hub(args.push_to_hub)
        logger.info("  → pushed successfully")

    logger.info("\nDone!  CodeQA is ready for Phase 2 training.")
    logger.info(
        "NAT will automatically load it from "
        f"'{args.output}' during Phase 2."
    )


if __name__ == "__main__":
    main()
