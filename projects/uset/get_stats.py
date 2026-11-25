from datasets import load_dataset
ds = load_dataset("mia/pick_place_16_success", split="train")
print(ds.stats)
