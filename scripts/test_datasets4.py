"""Get dataset sizes and final checks."""
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

def count_ds(name, loader):
    print(f"=== {name} count ===")
    try:
        ds = loader()
        n = 0
        for _ in ds:
            n += 1
            if n % 5000 == 0:
                print(f"  ...{n}")
        print(f"  Total: {n}\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")

# hendrycks_math with config
test_source(
    "EleutherAI/hendrycks_math (algebra)",
    lambda: load_dataset("EleutherAI/hendrycks_math", "algebra", split="train", streaming=True),
)

# coastalcph/lex_glue scotus - how many labels? what are they?
print("=== lex_glue scotus label distribution ===")
ds = load_dataset("coastalcph/lex_glue", "scotus", split="train", streaming=True)
labels = set()
n = 0
for x in ds:
    labels.add(x["label"])
    n += 1
    if n >= 2000:
        break
print(f"  Labels seen in first 2000: {sorted(labels)}")
print(f"  Count so far: {n}\n")

# Sizes for our final picks
count_ds("lighteval/MATH-Hard", lambda: load_dataset("lighteval/MATH-Hard", split="train", streaming=True))
count_ds("gsm8k (main)", lambda: load_dataset("gsm8k", "main", split="train", streaming=True))
