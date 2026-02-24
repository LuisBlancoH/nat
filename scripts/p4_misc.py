"""Check AR-LSAT test split and lex_glue scotus text quality."""
from datasets import load_dataset

# AR-LSAT — test split only
print("=== dmayhem93/agieval-lsat-ar (test split) ===")
try:
    ds = load_dataset("dmayhem93/agieval-lsat-ar", split="test", streaming=True)
    s = next(iter(ds))
    print(f"  keys={list(s.keys())}")
    for k, v in s.items():
        print(f"    {k}: {str(v)[:150]}")
except Exception as e:
    print(f"  FAILED: {e}")

print()

# lex_glue scotus — is text raw legal opinion or short excerpt?
print("=== lex_glue/scotus: text length + reasoning check ===")
ds2 = load_dataset("coastalcph/lex_glue", "scotus", split="train", streaming=True)
for i, ex in enumerate(ds2):
    print(f"  [ex {i}] label={ex['label']}  len={len(ex['text'])}  text[:300]={ex['text'][:300]!r}")
    if i >= 2:
        break

print()

# camel-ai/math — check sub_topic for grouping
print("=== camel-ai/math: topic distribution ===")
ds3 = load_dataset("camel-ai/math", split="train", streaming=True)
topics = {}
for i, ex in enumerate(ds3):
    t = ex.get("topic;", "?")  # note the semicolon in field name
    topics[t] = topics.get(t, 0) + 1
    if i >= 2000:
        break
print(f"  Unique topics in first 2000: {sorted(topics.keys())[:20]}")
print(f"  Total unique topics: {len(topics)}")
