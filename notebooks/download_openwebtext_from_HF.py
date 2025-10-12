import os
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Skylion007/openwebtext")

# Check what splits are available
print(dataset)

# Example: preview one text sample
print(dataset["train"][0]["text"])




os.makedirs("../data/openwebtext", exist_ok=True)

with open("../data/openwebtext/train.txt", "w", encoding="utf-8") as f:
    for item in dataset["train"]:
        f.write(item["text"] + "\n")