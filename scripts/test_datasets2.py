"""Test ScienceQA with Pillow, and find legalbench alternative."""
from datasets import load_dataset

def test_source(name, loader):
    print(f"=== {name} ===")
    try:
        ds = loader()
        sample = next(iter(ds))
        print(f"  Keys: {list(sample.keys())}")
        for k, v in sample.items():
            print(f"  {k}: {type(v).__name__} -> {str(v)[:100]}")
        print("  ✅ OK\n")
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")

# ScienceQA retry (Pillow should be installed now)
test_source(
    "derek-thomas/ScienceQA",
    lambda: load_dataset("derek-thomas/ScienceQA", split="train", streaming=True),
)

# LegalBench alternatives
test_source(
    "tasksource/legalbench",
    lambda: load_dataset("tasksource/legalbench", split="train", streaming=True),
)

test_source(
    "ricdomolmhemhem/legalbench-instruct",
    lambda: load_dataset("ricdomolmhemhem/legalbench-instruct", split="train", streaming=True),
)

test_source(
    "pile-of-law/pile-of-law (r_legaladvice)",
    lambda: load_dataset("pile-of-law/pile-of-law", "r_legaladvice", split="train", streaming=True),
)
