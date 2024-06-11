from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F 

#---------

class CasualSelfAttention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_emb % config.n_head == 0
        #key, query, value projections for all heads, but in batch
        self.c_attn = nn.Linear(config.n_emb, 3* config.n_emb)
        #output projection
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        # regulirazation
        self.n_head = config.n_head
        self.n_emb = config.n_emb
        #following the naming conv of HF/OPENAI aka CLOSEDAI, more of a mask not really bias
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() #batch size, embedding length, embedding dimentionality(n_emb)
        #calculate key, query, value for all heads in a batch and move head forward to be the batch
        #nh is the num of heads, 'hs' is the head size, C is the number of channels = nh * hs
        #e.g in GPT2(124M) n_head = 12, hs= 64, nh*hs=C=768 channels in the transformers
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_emb, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        #attention (materilizes the large(T,T)matrix for all queries and key)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v #(B, nh, T, T) x (B,nh, T, hs) => (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)# re-assemble all head outputs side by side, this also computes the concatation
        #output projection
        y = self.c_proj(y)
        return y 
        


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, 4* config.n_emb)
        self.gelu = nn.GELU(approximate='tanh') # this was first historically used (in GPT2 as well), but now the exact gelu can be used
        self.c_proj = nn.Linear(4* config.n_emb, config.n_emb)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class  Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.MLP = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.MLP(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 #max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE Merges + 256 bytes tokens + 1 <|endoftext|>token
    n_layer: int = 12 # number of layers
    n_head: int = 12 #  number of heads
    n_emb:int = 768 # embedding dimention

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb),
            wpe = nn.Embedding(config.block_size, config.n_emb),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_emb),
        ))
        #GPT2 paper no bias
        self.ln_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)

