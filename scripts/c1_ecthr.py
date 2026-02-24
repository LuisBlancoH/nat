"""Check lex_glue/ecthr_a for reasoning traces — look at 5 full examples."""
from datasets import load_dataset

ds = load_dataset("coastalcph/lex_glue", "ecthr_a", split="train", streaming=True)
for i, ex in enumerate(ds):
    print(f"\n--- Example {i} ---")
    print(f"labels: {ex['labels']}")
    text = ex['text']
    # text is a list of paragraphs
    print(f"num paragraphs: {len(text)}")
    print(f"para[0]: {text[0][:300]!r}")
    if len(text) > 1:
        print(f"para[-1]: {text[-1][:300]!r}")
    if i >= 4:
        break
