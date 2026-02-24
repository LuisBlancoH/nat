"""Check PrOntoQA and AR-LSAT candidates."""
from datasets import load_dataset

candidates = [
    ("sihaochen/PrOntoQA", None),
    ("dmayhem93/agieval-lsat-ar", None),
    ("hails/agieval-lsat-ar", None),
    ("zhengxiangshi/AR-LSAT", None),
]

for name, cfg in candidates:
    try:
        kw = {"split": "train", "streaming": True}
        if cfg:
            kw["name"] = cfg
        ds = load_dataset(name, **kw)
        s = next(iter(ds))
        print(f"✅ {name}  keys={list(s.keys())}")
        for k, v in s.items():
            print(f"    {k}: {str(v)[:120]}")
    except Exception as e:
        print(f"❌ {name}: {e}")
    print()
