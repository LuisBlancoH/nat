from datasets import load_dataset
s = next(iter(load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)))
print(list(s.keys()), str(s)[:400])
