import enum
import os
import random
from tqdm import tqdm
from datasets import load_from_disk


load_dataset = load_from_disk("./openwebtext_local")
# output_file = "output{}.txt"
vocab_file = "vocab.txt"

split_files = 1          # how many separate .txt files you want
split_size = len(load_dataset['train']) // split_files
vocab = set()

for i in range(split_files):
    start = i * split_size
    end = (i + 1) * split_size if i < split_files - 1 else len(load_dataset['train'])
    out_path = f"../data/openwebtext/openwebtext_part_{i}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        for item in tqdm(load_dataset['train'][start:end], desc=f"File {i+1}/{split_files}"):
            text = item["text"]
            if text:
                f.write(text.replace("\n", " ") + "\n")
                vocab.update(text)

with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')


# Define file paths
original_training_file = "../data/train_split.txt"
original_validation_file = "../data/val_split.txt"

# Create 1GB sample files for trial
sample_training_file = "../data/train_sample_1Gb.txt"
sample_validation_file = "../data/val_sample_100MB.txt"

# Create 1GB sample from training file
print("Creating 1GB training sample...")
with open(original_training_file, "rb") as f:
    chunk = f.read(1 * 1024**3)  # Read first 1GB

with open(sample_training_file, "wb") as f:
    f.write(chunk)
print(f"✓ Created {sample_training_file}")

# Create 1GB sample from validation file (optional, or make it smaller)
print("Creating validation sample...")
with open(original_validation_file, "rb") as f:
    chunk = f.read(100 * 1024**2)  # Read 100MB for validation

with open(sample_validation_file, "wb") as f:
    f.write(chunk)
print(f"✓ Created {sample_validation_file}")




def get_random_chunk(split):
    filename = sample_training_file if split == "train" else sample_validation_file
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            #file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            #seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            #dcode the block to string, ignoring any invalid byte sequences
            decoded_block = block.decode("utf-8", errors="ignore").replace('\r', '')

            #Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size]for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y