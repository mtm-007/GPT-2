import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random

from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import GradScaler, autocast
# from torch.cuda.amp.grad_scaler import GradScaler
# from torch.cuda.amp.autocast_mode import autocast

from Transformer_functions import Head, MultiHeadAttention, FeedForward, Block, GPTLanguagemodel
from estimate_loss import estimate_loss
from data_batching_and_chunk_data import get_batch


import wandb
wandb.login()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.manual_seed(1337)

block_size = 512
batch_size = 32
max_iters = 2
learning_rate = 3e-4
eval_iters = 1
eval_interval = 1
n_embed = 384
n_layer = 4
n_head = 4
dropout = 0.2

# wandb tracking initialization

wandb.init(project = 'nano-gpt-tracking-test',
      config={
            "block_size" : 512,
            "batch_size" :32,
            "max_iters" :2,
            "eval_iterval" : 1,
            "lr" : 3e-4,
            "eval_iters" : 1,
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
print(f"Text size in memory: {size_mb:.2f} MB")

strng_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_strng = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [strng_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_strng[i] for i in l])



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

    for iter in range(max_iters):
        # --- Evaluation & logging ---
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            wandb.log({
                "step": iter,
                "train_loss": losses["train"],
                "val_loss": losses["val"]
            })

        xb, yb = get_batch("train")

        # --- Profiling for first few steps ---
        if iter < profile_steps:
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

            # --- Profiling summary ---
            print("\n=============")
            print(f"Profiling Summary (Step {iter})")
            print("=============")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            wandb.log({
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
