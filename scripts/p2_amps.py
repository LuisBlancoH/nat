"""Check AMPS candidates."""
from datasets import load_dataset
for name, cfg in [
    ("math-ai/amps_khan_academy", None),
    ("camel-ai/math", None),
    ("hendrycks/math", None),
]:
    try:
        kw = {"split": "train", "streaming": True}
        if cfg:
            kw["name"] = cfg
        ds = load_dataset(name, **kw)
        s = next(iter(ds))
        print(f"✅ {name}  keys={list(s.keys())}  sample={str(s)[:200]}")
    except Exception as e:
        print(f"❌ {name}: {e}")
