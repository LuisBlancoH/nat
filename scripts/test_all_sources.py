"""End-to-end test: load each source from the updated SOURCES list."""
import sys
sys.path.insert(0, "/Users/luis/Developer/nat")

from datasets import load_dataset

# Reproduce exactly what MultiDomainEpisodeDataset.SOURCES contains
SOURCES = [
    {"name": "gsm8k", "config": "main", "domain": "math", "split": "train"},
    # hendrycks_math 7 configs
    *[
        {"name": "EleutherAI/hendrycks_math", "config": cfg, "domain": "math", "split": "train"}
        for cfg in ["algebra", "counting_and_probability", "geometry",
                     "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    ],
    {"name": "lighteval/MATH-Hard", "config": None, "domain": "math_hard", "split": "train"},
    {"name": "open-r1/codeforces-cots", "config": None, "domain": "code", "split": "train"},
    {"name": "coastalcph/lex_glue", "config": "scotus", "domain": "logic", "split": "train"},
    {"name": "ucinlp/drop", "config": None, "domain": "reading", "split": "train"},
    {"name": "derek-thomas/ScienceQA", "config": None, "domain": "science", "split": "train"},
]

ok = 0
fail = 0
for src in SOURCES:
    label = f"{src['name']}" + (f" ({src['config']})" if src.get('config') else "")
    try:
        kwargs = {"split": src["split"], "streaming": True}
        if src.get("config"):
            kwargs["name"] = src["config"]
        ds = load_dataset(src["name"], **kwargs)
        sample = next(iter(ds))
        print(f"  ✅ {label:50s}  keys={list(sample.keys())[:5]}")
        ok += 1
    except Exception as e:
        print(f"  ❌ {label:50s}  {e}")
        fail += 1

print(f"\n{ok}/{ok+fail} sources OK, {fail} failed")
