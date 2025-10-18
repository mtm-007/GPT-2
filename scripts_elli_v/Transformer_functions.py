import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random


block_size = 512
batch_size = 8
max_iters = 3001
learning_rate = 3e-4
eval_iters = 200
eval_interval = 200
n_embed = 768
n_layer = 12
n_head = 12
dropout = 0.2

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
            #nn.ReLU(),
            nn.GELU(),
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

