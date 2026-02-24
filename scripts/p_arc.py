from datasets import load_dataset
s = next(iter(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", streaming=True)))
print(list(s.keys()), str(s)[:400])
