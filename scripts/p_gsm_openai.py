from datasets import load_dataset
s = next(iter(load_dataset("openai/gsm8k", "main", split="train", streaming=True)))
print(list(s.keys()), str(s)[:400])
