"""
Probe new/updated dataset sources for NAT.
Run: conda run -n qwen python scripts/probe_sources.py
"""
from datasets import load_dataset

def probe(label, fn):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)
    try:
        result = fn()
        print(result)
    except Exception as e:
        print(f"  FAILED: {e}")


# ── 1. CodeForces-CoTs: does it have algorithm tags? ──────────────────
def check_codeforces_tags():
    ds = load_dataset("open-r1/codeforces-cots", split="train", streaming=True)
    samples = [next(iter(ds)) for _ in range(1)]
    # Check all keys
    sample = samples[0]
    out = [f"  ALL KEYS: {list(sample.keys())}"]
    for k in ["tags", "difficulty", "category", "algorithm", "topic"]:
        if k in sample:
            out.append(f"  FOUND '{k}': {sample[k]}")
        else:
            out.append(f"  missing '{k}'")
    return "\n".join(out)

probe("codeforces-cots: algorithm tags?", check_codeforces_tags)


# ── 2. TACO dataset (BAAI/TACO) — algorithm labels ────────────────────
def check_taco():
    ds = load_dataset("BAAI/TACO", split="train", streaming=True)
    sample = next(iter(ds))
    out = [f"  Keys: {list(sample.keys())}"]
    for k, v in sample.items():
        out.append(f"  {k}: {type(v).__name__} -> {str(v)[:120]}")
    return "\n".join(out)

probe("BAAI/TACO", check_taco)


# ── 3. AMPS alternatives ───────────────────────────────────────────────
for name in ["math-ai/amps_khan_academy", "keirp/amps", "camel-ai/math"]:
    def check_amps(n=name):
        ds = load_dataset(n, split="train", streaming=True)
        sample = next(iter(ds))
        return f"  Keys: {list(sample.keys())}\n  sample: {str(sample)[:300]}"
    probe(f"AMPS candidate: {name}", check_amps)


# ── 4. PrOntoQA ────────────────────────────────────────────────────────
for name in ["prosyslab/prontoqa", "zhuynz/prontoqa", "sihaochen/PrOntoQA"]:
    def check_prontoqa(n=name):
        ds = load_dataset(n, split="train", streaming=True)
        sample = next(iter(ds))
        return f"  Keys: {list(sample.keys())}\n  sample: {str(sample)[:300]}"
    probe(f"PrOntoQA candidate: {name}", check_prontoqa)


# ── 5. AR-LSAT ─────────────────────────────────────────────────────────
for name in ["zhengxiangshi/AR-LSAT", "dmayhem93/agieval-lsat-ar",
             "hails/agieval-lsat-ar"]:
    def check_lsat(n=name):
        ds = load_dataset(n, split="train", streaming=True)
        sample = next(iter(ds))
        return f"  Keys: {list(sample.keys())}\n  sample: {str(sample)[:300]}"
    probe(f"AR-LSAT candidate: {name}", check_lsat)


# ── 6. Lex_glue SCOTUS: what does the text look like? ─────────────────
def check_scotus():
    ds = load_dataset("coastalcph/lex_glue", "scotus", split="train", streaming=True)
    samples = []
    for i, ex in enumerate(ds):
        samples.append(ex)
        if i >= 2:
            break
    out = []
    for s in samples:
        out.append(f"  label={s['label']}  text_preview={s['text'][:200]!r}")
    return "\n".join(out)

probe("lex_glue/scotus: text quality check", check_scotus)
