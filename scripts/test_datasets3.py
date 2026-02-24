"""Search for working legal reasoning datasets."""
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

test_source(
    "casehold/casehold",
    lambda: load_dataset("casehold/casehold", split="train", streaming=True),
)

test_source(
    "coastalcph/lex_glue (scotus)",
    lambda: load_dataset("coastalcph/lex_glue", "scotus", split="train", streaming=True),
)

test_source(
    "coastalcph/lex_glue (ecthr_a)",
    lambda: load_dataset("coastalcph/lex_glue", "ecthr_a", split="train", streaming=True),
)

test_source(
    "joelito/lextreme (swiss_judgment_prediction)",
    lambda: load_dataset("joelito/lextreme", "swiss_judgment_prediction", split="train", streaming=True),
)

# AMPS replacement: maybe a khan academy style dataset
test_source(
    "EleutherAI/hendrycks_math",
    lambda: load_dataset("EleutherAI/hendrycks_math", split="train", streaming=True),
)

test_source(
    "gsm8k (main)",
    lambda: load_dataset("gsm8k", "main", split="train", streaming=True),
)
