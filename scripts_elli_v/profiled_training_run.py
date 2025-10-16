import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap, random, time
import statistics

from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import GradScaler, autocast
# from torch.cuda.amp.grad_scaler import GradScaler
# from torch.cuda.amp.autocast_mode import autocast

from Transformer_functions import Head, MultiHeadAttention, FeedForward, Block, GPTLanguagemodel
from data_batching_and_chunk_data import get_batch, PrefetchLoader


import wandb
wandb.login()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.manual_seed(1337)

block_size = 512
batch_size = 32*2
max_iters = 1000
learning_rate = 3e-4
eval_iters = 100
eval_interval = 100
n_embed = 384
n_layer = 4
n_head = 4
dropout = 0.2

# wandb tracking initialization

wandb.init(project = 'nano-gpt-tracking-test',
      config={
            "block_size" : 512,
            "batch_size" :32*2,
            "max_iters" :1000,
            "eval_iterval" : 100,
            "lr" : 3e-4,
            "eval_iters" : 100,
            "n_emb" : 384,
            "n_layer" : 4,
            "n_head" :4,
            "dropout" : 0.2,
            #"dtype" : 'bfloat16' didnt work yet need some debugging,
}
)

chars = ""
data_used = '../data/vocab.txt'
with open(data_used, 'r', encoding='utf-8')as f:
    text = f.read()
    chars = sorted(set(text))

vocab_size = len(chars)

size_mb = sys.getsizeof(text) / (1024 * 1024)
print(f"Vocab_size  in memory: {size_mb:.2f} MB")

strng_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_strng = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [strng_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_strng[i] for i in l])

train_loader = PrefetchLoader(split="train", prefetch=8)
val_loader   = PrefetchLoader(split="val", prefetch=2)

@torch.no_grad()
def estimate_loss():
    """
    OPTIMIZED: Keep losses on GPU, single sync at end
    """
    out = {}
    model.eval()
    for split in ['train','val']:
        # FIXED: Create tensor on GPU
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            #X,Y = get_batch(split)
            if split == "train":
                X, Y = train_loader.next()
            else:
                X, Y = val_loader.next()
            logits, loss = model(X,Y)
            # FIXED: No .item() - keep on GPU
            losses[k] = loss
        # FIXED: Single GPU->CPU sync at the end
        out[split] = losses.mean().item()
    model.train()
    return out


model = GPTLanguagemodel(vocab_size).to(device)
#add torch compile 
m = torch.compile(model)

wandb.watch(m)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


#lets try without AMP

# OPTION A: Remove AMP completely (try this first)
def train_model(model=model, 
                optimizer=optimizer, 
                max_iters=max_iters, 
                eval_interval=eval_interval, 
                profile_steps=3, 
                USE_AMP=False):
                #get_batch, 
                #estimate_loss)
    """
    Train a model with optional AMP and profiling.
    """
    start_time = datetime.now()
    print(f"\nTraining started at {start_time.strftime('%H:%M:%S')}")
    print(f"AMP enabled: {USE_AMP}")

    scaler = GradScaler() if USE_AMP else None

    # --- Lists to collect data vs GPU times ---
    data_times, gpu_times = [], []

    for iter in range(max_iters):
        # --- Evaluation & logging ---
        if iter % eval_interval == 0:

            data_times, gpu_times = [], []

            losses = estimate_loss()
            print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            
            wandb.log({
                "step": iter,
                "train_loss": losses["train"],
                "val_loss": losses["val"]
            })
        

        #xb, yb = get_batch("train")
        #xb, yb = train_loader.next()


        # ------------------------------
        # TIMING: Data loading vs GPU compute
        # ------------------------------
        torch.cuda.synchronize()
        t0 = time.time()
        xb, yb = train_loader.next()  # preloaded async batch
        torch.cuda.synchronize()
        t1 = time.time()

        # --- Profiling for first few steps ---
        if iter < profile_steps:
            print("\n=== GPU Memory Summary ===")
            print(torch.cuda.memory_summary(device=device, abbreviated=False))

            with profile(use_cuda=torch.cuda.is_available()) as prof:
                with record_function(f"train_step_{iter}"):
                    optimizer.zero_grad(set_to_none=True)

                    if USE_AMP:
                        with autocast():
                            logits, loss = model(xb, yb)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits, loss = model(xb, yb)
                        loss.backward()
                        optimizer.step()

            torch.cuda.synchronize()
            t2 = time.time()

            # data_load_time = t1 - t0
            # gpu_compute_time = t2 - t1
            # Record times
            data_times.append(t1 - t0)
            gpu_times.append(t2 - t1)
            #print(f"Iter {iter}: data load {data_load_time:.4f}s, gpu compute {gpu_compute_time:.4f}s")
            #print(f"data load {data_load_time:.4f}s, gpu compute {gpu_compute_time:.4f}s")

            # --- Profiling summary ---
            print("\n=============")
            print(f"Profiling Summary (Step {iter})")
            print("=============")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            wandb.log({
                # "data_load_time": data_load_time,
                # "gpu_compute_time": gpu_compute_time,
                f"profiler/step_{iter}_table": wandb.Html(
                    "<pre>" + prof.key_averages().table(sort_by="cuda_time_total", row_limit=10) + "</pre>"
                )
            })

        else:
            # --- Regular training (no profiling) ---
            optimizer.zero_grad(set_to_none=True)

            if USE_AMP:
                with autocast():
                    logits, loss = model(xb, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = model(xb, yb)
                loss.backward()
                optimizer.step()
    
            torch.cuda.synchronize()
            t2 = time.time()

            # data_load_time = t1 - t0
            # gpu_compute_time = t2 - t1
            data_times.append(t1 - t0)
            gpu_times.append(t2 - t1)


        if iter % eval_interval == 0:
            # print("\n=== GPU Memory Summary ===")
            # print(torch.cuda.memory_summary(device=device, abbreviated=False))

            if data_times and gpu_times:
                print("\n=== Average Data vs GPU Time Summary ===")
                print(f"Data load time: mean={statistics.mean(data_times):.4f}s, "
                    f"median={statistics.median(data_times):.4f}s, "
                    f"mode={statistics.mode(data_times):.4f}s")
                print(f"GPU compute time: mean={statistics.mean(gpu_times):.4f}s, "
                    f"median={statistics.median(gpu_times):.4f}s, "
                    f"mode={statistics.mode(gpu_times):.4f}s")
                print("\n--------------data load time------------")

                wandb.log({
                    "data_load_mean": statistics.mean(data_times),
                    "data_load_median": statistics.median(data_times),
                    "data_load_mode": statistics.mode(data_times),
                    "gpu_compute_mean": statistics.mean(gpu_times),
                    "gpu_compute_median": statistics.median(gpu_times),
                    "gpu_compute_mode": statistics.mode(gpu_times)
                })


            #print(f"Iter {iter}: data load {data_load_time:.4f}s, gpu compute {gpu_compute_time:.4f}s")
            # print(f"data load {data_load_time:.4f}s, gpu compute {gpu_compute_time:.4f}s")
            # wandb.log({
            #     #"step": iter,
            #     "data_load_time": data_load_time,
            #     "gpu_compute_time": gpu_compute_time,
            # })


    torch.cuda.empty_cache() ## unnecessary sync overhead

    # --- Final Summary ---
    end_time = datetime.now()
    total_minutes = (end_time - start_time).total_seconds() / 60

    print(f"\nTraining completed at {end_time.strftime('%H:%M:%S')}")
    if loss is not None:
        print(f"Final loss: {loss.item():.4f}")
    print(f"Total training time (CPU + GPU overhead): {total_minutes:.2f} minutes")

    wandb.log({
        "final_loss": loss.item() if loss is not None else None,
        "total_training_minutes": total_minutes
    })

train_model(
    model=model,
    optimizer=optimizer,
    max_iters=max_iters,        # total training iterations
    eval_interval=eval_interval,     # print & log every 50 iterations
    profile_steps=3,      # profile first 3 steps
    USE_AMP=True
    # get_batch=get_batch,
    # estimate_loss=estimate_loss,         # enable mixed precision
)

# Generation
context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)
