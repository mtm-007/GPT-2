import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap,threading, queue, random

from pathlib import Path
from datetime import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.manual_seed(1337)


block_size = 512
batch_size = 8


chars = ""
data_used = '../data/vocab.txt'
with open(data_used, 'r', encoding='utf-8')as f:
    text = f.read()
    chars = sorted(set(text))

vocab_size = len(chars)
print(f"Vocab size used for training: {vocab_size:.2f}")


strng_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_strng = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [strng_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_strng[i] for i in l])


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
#print(f"✓ Created {sample_training_file}")

# Create 1GB sample from validation file (optional, or make it smaller)

#print("Creating validation sample...")
with open(original_validation_file, "rb") as f:
    chunk = f.read(100 * 1024**2)  # Read 100MB for validation

with open(sample_validation_file, "wb") as f:
    f.write(chunk)
#print(f"✓ Created {sample_validation_file}")


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
    #x, y = x.to(device), y.to(device)
    #return x,y
    return x.pin_memory(), y.pin_memory()

class PrefetchLoader:
    def __init__(self, split="train", prefetch=4, num_threads=4):
        self.split = split
        self.queue = queue.Queue(maxsize=prefetch)
        # self.thread = threading.Thread(target=self._worker, daemon=True)
        # self.thread.start()
        self.threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.threads.append(t)

    def _worker(self):
        while True:
            batch = get_batch(self.split)
            self.queue.put(batch)

    def next(self):
        # Wait for the next preloaded batch
        x, y = self.queue.get()
        # Move asynchronously to GPU
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        return x, y
