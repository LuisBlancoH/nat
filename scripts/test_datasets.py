"""Quick test: can we actually fetch each dataset source?"""
from datasets import load_dataset

def test_source(name, loader, desc=""):
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

# --- Already confirmed broken ---
# EleutherAI/amps (khan_academy) -> does not exist
# hendrycks/competition_math -> does not exist

# --- Already confirmed working ---
# open-r1/codeforces-cots -> OK

# --- Need to test ---
test_source(
    "ucinlp/drop",
    lambda: load_dataset("ucinlp/drop", split="train", streaming=True),
)

test_source(
    "derek-thomas/ScienceQA",
    lambda: load_dataset("derek-thomas/ScienceQA", split="train", streaming=True),
)

test_source(
    "nguha/legalbench (script-based, likely broken)",
    lambda: load_dataset("nguha/legalbench", split="train", streaming=True),
)

# --- MATH replacement candidate ---
test_source(
    "lighteval/MATH-Hard",
    lambda: load_dataset("lighteval/MATH-Hard", split="train", streaming=True),
)
