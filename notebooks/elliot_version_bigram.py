#!/usr/bin/env python
# coding: utf-8

import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import GradScaler, autocast

import wandb
wandb.login()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.manual_seed(1337)

block_size = 1024
batch_size = 128
max_iters = 201
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
            "block_size" : 1024,
            "batch_size" :128,
            "max_iters" :201,
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
data_used = '../data/input.txt'
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

# encoded_hello = torch.tensor(encode('hello'),dtype=torch.long)
# decoded_hello = decode(encoded_hello.tolist())

data = torch.tensor(encode(text), dtype=torch.long)


n = int(0.9*len(data))

train_data = data[:n]
val_data = data[n:]

train_data = train_data.to(device, non_blocking=True)
val_data = val_data.to(device, non_blocking=True)


def get_batch(split):
    """
    Efficient GPU-native batch sampling for sequence data.
    Returns x, y already on the GPU, contiguous.
    """
    data = train_data if split == 'train' else val_data
    # Sample batch indices directly on GPU
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)

    # Create a 2D tensor of shape (batch_size, block_size) for x
    x = data[ix[:, None] + torch.arange(block_size, device=device)]
    y = data[ix[:, None] + torch.arange(1, block_size + 1, device=device)]

    # Optional: make contiguous for better memory access in CUDA
    return x.contiguous(), y.contiguous()


# def get_batch(split):
#     data = train_data if split=='train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     #print(ix)
#     x = data[ix[:, None] + torch.arange(block_size, device=device)]
#     y = data[ix[:, None] + torch.arange(1, block_size + 1, device=device)]
#     return x, y
#     #x, y = x.to(device), y.to(device)
#     #return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

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
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            # FIXED: No .item() - keep on GPU
            losses[k] = loss
        # FIXED: Single GPU->CPU sync at the end
        out[split] = losses.mean().item()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        #input size(Batch, time-step, channels)
        #output size(Batch, time-step, headsize)
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_s)
        q = self.query(x) #(B,T,head_s)
        #compute attention scores, ('affinities')
        wei = q @ k.transpose(-2,-1) * k.shape[-1]** -0.5 #(B,T,head_s) @ (B, head_s,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] ==0,float('-inf')) #(B, T, T)
        wei = F.softmax(wei, dim=-1) #(B, T, T)
        wei = self.dropout(wei)
        #perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,head_s)
        out = wei @ v #(B,T,T) @ (B,T,head_s) -> (B,T,head_s)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel """
    
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embed) # bringing back n_embed from: head_size = n_embed // n_head
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, C(feautures like in head)
        out = self.dropout(self.proj(out))
        return out
        

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # 4 is hyperparameter, Expanding dimentional for learning..hidden dim of FFN is 4 times (sweetspot)
            nn.ReLU(),
            nn.Linear(4* n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)
        

class Block(nn.Module):
    """ Transformer block: communication followed by computation"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head #num of features(dimention)  captured in each MHA
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed) # postnorm
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        """ here y is not target, after computation on x so later x can be added as resnet"""
        #post-norm like attention paper
        y = self.sa(x)
        x = self.ln1(x+y) # the Residual part here just added the previuos x ..y(x) + x(resnet)
        y = self.ffwd(x)
        x = self.ln2(x+y)
        return x


class GPTLanguagemodel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)# vocab_size)
        self.positional_embedding_table = nn.Embedding(block_size, n_embed)# vocab_size)

        # FIXED: Pre-create position indices buffer
        self.register_buffer('position_ids', torch.arange(0, block_size, dtype=torch.long))

        
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, index, targets=None):
        B,T = index.shape
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"
        
        tok_emb = self.token_embedding_table(index) # (B, T, C)
        #pos_emb = self.positional_embedding_table(torch.arange(0,T, device=device))# (T, C)
        # FIXED: Use pre-created buffer, just slice what we need
        pos_emb = self.positional_embedding_table(self.position_ids[:T])
        x = tok_emb + pos_emb #(B,T, C)
        x = self.blocks(x) #(B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # ---- Compute loss only if targets are provided (training mode) ----
        if targets is None:
            loss = None
        else:
            # Flatten both tensors to feed into cross_entropy
            B,T,C = logits.shape  #T "time" is the sequence size or block_size, C "channel" is vocab_size
            logits = logits.view(B*T, C) # B*T act as total number of samples, C class scores
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #cross_Entropy expects input:(N,C), targets:(N,)
            
        return logits,loss

    def generate(self, index, max_new_tokens):
        #index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #get predictions
            #cropping the context ( idx) to the last block_size tokens, otherwise our posi_emb will run out of scope
            idx_cond = index[:, -block_size:]
            logits, loss = self.forward(idx_cond)
            #for generation (not training) targets is None so skips logits flattening -> logtis become (B,T,C)
            logits = logits[:,-1,:] #becomes (B, C), -1 gets only the last token from the time step
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C), dim=-1 acts across C class channels
            #sample for distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1), concatenates across time sequence, if dim=0 it would stack batches
        return index


model = GPTLanguagemodel(vocab_size).to(device)
#add torch compile 
m = torch.compile(model)

wandb.watch(m)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#lets try without AMP

# OPTION A: Remove AMP completely (try this first)
USE_AMP = False

if USE_AMP:
    scaler = GradScaler()
    print("Using Automatic Mixed Precision (AMP)")
else:
    print("Using pure FP32 (no AMP)")

start_time = datetime.now()

# Profile only limited steps
profile_steps = 3

for iter in range(max_iters):
    # Evaluation & WandB logging
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        wandb.log({
            "step": iter,
            "train_loss": losses["train"],
            "val_loss": losses["val"]
        })

    xb, yb = get_batch("train")

    # Profiling only for first few steps
    if iter < profile_steps:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False
        ) as prof:
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

        # Concise summary with copy detection
        events = prof.key_averages()
        total_cuda_time = sum(e.cuda_time_total for e in events)
        total_copy = sum(e.cuda_time_total for e in events if 'copy' in e.key.lower())
        top_ops = sorted(events, key=lambda e: e.cuda_time_total, reverse=True)[:5]

        print(f"\n---- Profiler Summary (Step {iter}) ----")
        for op in top_ops:
            print(f"{op.key:<40} {op.cuda_time_total/1000:.2f} ms")
        print(f"Total CUDA time: {total_cuda_time/1000:.2f} ms")
        print(f"Total copy time: {total_copy/1000:.2f} ms ({100*total_copy/total_cuda_time:.1f}%)")

        wandb.log({
            f"profiler/step_{iter}_cuda_ms": total_cuda_time / 1000,
            f"profiler/step_{iter}_copy_ms": total_copy / 1000,
            f"profiler/top_ops_step_{iter}": wandb.Html(
                "<pre>" + "\n".join([f"{op.key}: {op.cuda_time_total/1000:.2f} ms" for op in top_ops]) + "</pre>"
            )
        })

    else:
        # Regular training (no profiling overhead)
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

    # REMOVED: torch.cuda.empty_cache() - unnecessary sync overhead

# Training summary
print(f"\nFinal loss: {loss.item():.4f}")

end_time = datetime.now()
total_minutes = (end_time - start_time).total_seconds() / 60
print(f"\nTotal training time: {total_minutes:.2f} minutes")

wandb.log({"total_training_minutes": total_minutes})

# Generation
context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)






# scaler = GradScaler()

# start_time = datetime.now()

# # Profile only limited steps (avoid big overhead)
# profile_steps = 3  # or 50 if you really need longer profiling

# for iter in range(max_iters):
#     # ---- Evaluation & WandB logging ----
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
#         wandb.log({
#             "step": iter,
#             "train_loss": losses["train"],
#             "val_loss": losses["val"]
#         })

#     xb, yb = get_batch("train")

#     # ---- Profiling only for first few steps ----
#     if iter < profile_steps:
#         with profile(
#             activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#             record_shapes=False,
#             profile_memory=False,
#             with_stack=False
#         ) as prof:
#             with record_function(f"train_step_{iter}"):
#                 optimizer.zero_grad(set_to_none=True)
#                 with autocast():
#                     logits, loss = model(xb, yb)
#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 scaler.update()

#         # ---- Concise summary ----
#         events = prof.key_averages()
#         total_cuda_time = sum(e.cuda_time_total for e in events)
#         top_ops = sorted(events, key=lambda e: e.cuda_time_total, reverse=True)[:5]

#         print(f"\n---- Profiler Summary (Step {iter}) ----")
#         for op in top_ops:
#             print(f"{op.key:<40} {op.cuda_time_total/1000:.2f} ms")
#         print(f"Total CUDA time: {total_cuda_time/1000:.2f} ms")

#         # ---- Log to wandb (short summary) ----
#         wandb.log({
#             f"profiler/step_{iter}_cuda_ms": total_cuda_time / 1000,
#             f"profiler/top_ops_step_{iter}": wandb.Html(
#                 "<pre>" + "\n".join([f"{op.key}: {op.cuda_time_total/1000:.2f} ms" for op in top_ops]) + "</pre>"
#             )
#         })

#         # Optional trace export for Chrome/TensorBoard
#         # prof.export_chrome_trace(f"trace_step_{iter}.json")

#     else:
#         # ---- Regular training ----
#         optimizer.zero_grad(set_to_none=True)
#         with autocast():
#             logits, loss = model(xb, yb)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#     torch.cuda.empty_cache()

# # ---- Training summary ----
# print(f"\nFinal loss: {loss.item():.4f}")

# end_time = datetime.now()
# total_minutes = (end_time - start_time).total_seconds() / 60
# print(f"\nTotal training time: {total_minutes:.2f} minutes")

# wandb.log({"total_training_minutes": total_minutes})



# #generation
# context = torch.zeros((1,1), dtype=torch.long, device=device)
# generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
# print(generated_chars)



