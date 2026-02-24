"""Check TACO for algorithm tags."""
from datasets import load_dataset
ds = load_dataset("BAAI/TACO", split="train", streaming=True)
s = next(iter(ds))
print("Keys:", list(s.keys()))
for k, v in s.items():
    print(f"  {k}: {str(v)[:150]}")
