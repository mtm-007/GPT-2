from dataclasses import dataclass
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F 

import wandb
wandb.login()

from utils import DataLoaderLite

#---------


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #key, query, value projections for all heads, but in batch
        self.c_attn = nn.Linear(config.n_embd, 3* config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        #andrej Karpathy_special_edition scaling residual
        self.c_proj.NANOGPT_SCALE_INIT=1
        # regulirazation
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #following the naming conv of HF/OPENAI aka CLOSEDAI, more of a mask not really bias
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() #batch size, embedding length, embedding dimentionality(n_embd)
        #calculate key, query, value for all heads in a batch and move head forward to be the batch
        #nh is the num of heads, 'hs' is the head size, C is the number of channels = nh * hs
        #e.g in GPT2(124M) n_head = 12, hs= 64, nh*hs=C=768 channels in the transformers
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        #attention (materilizes the large(T,T)matrix for all queries and key)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v #(B, nh, T, T) x (B,nh, T, hs) => (B, nh, T, hs)

        #flash attention implemented in pytorch
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)# re-assemble all head outputs side by side, this also computes the concatation
        #output projection
        y = self.c_proj(y)
        return y 
        


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4* config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # this was first historically used (in GPT2 as well), but now the exact gelu can be used
        self.c_proj = nn.Linear(4* config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 #max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE Merges + 256 bytes tokens + 1 <|endoftext|>token
    n_layer: int = 12 # number of layers
    n_head: int = 12 #  number of heads
    n_embd:int = 768 # embedding dimention

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        #GPT2 paper no bias
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight sharing/tying scheme, this alone saves 1/3 of the parameters size in GPT2(124) as (768*50257~38.5 million Para),also improves perf
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std=0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)#this GPT2 initialization is roughly equivalent to the xavier initialazation too
            if module.bias is not None:
                #by default in pytorch bias is initialized with a uniform
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets = None):
        #idx is in a shape of (B,T), T is upto the block size( max sequence length)
        B, T = idx.size() # (B by T) is 2 dim tensor of T token rows then staked rows as B batchs 
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        #this is going to define the elements one by one following the GPT __init__(initialization) in the self.transformer -> (wte,wpe,h,ln_f, then lm_head)
        #forward the token and positional embeddings
        pos = torch.arange(0 , T, dtype= torch.long, device = idx.device) #shape(T)
        pos_emb = self.transformer.wpe(pos) #positional embedding of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) #token embedding of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        #forward block of the transformer
        for block in self.transformer.h:
            x = block(x)
        #forwars the final layer norm in the tranformer
        x = self.transformer.ln_f(x)
        #every (B,T) 2 dim tensors calculates logits what comes next
        logits = self.lm_head(x) #(B, T, vocab_size), vocab size is the number of possible tokens, looking for (B, T+1 token)
        loss = None
        if targets is not None:
            #cross entropy does not like multidim inputs so we flatten them from 3 dim to 2 dim 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'} #gpt2 is the 125M, and gpt2-xl is the 1.5B
        from transformers import GPT2LMHeadModel 
        print("loading gpt2 weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd =768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer =48, n_head =25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        #create a from scratch 
        config = GPTConfig(**config_args)
        model = GPT(config)
        #creating stat_dict for both our model and for weight from huggingface
        sd = model.state_dict()
        sd_keys = sd.keys()
        # ignoring the buffer(they are not parameters) biases that come with the autoregressive mask
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
# ------------------------------------------------------------------------------------
# wandb tracking initialization
wandb.init(project = 'nano-gpt-tracking-test',
      config={
            "B" :16,
            "T" : 1024,
            "lr" : 3e-4,
            "iterations" :50*20,
            "torch compile": "True",
            "flash_attention": "True",
            "vocab_size_padded_to_even_num": "True", 

      })

import time

device= "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device= "mps"
#device = "cpu" #overide
print(f"the available device is: {device}")
wandb.log({"the available device is": str(device)})

def device_synchronise():
    '''autoselect torch synchronization based on device available'''
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
# Print device and device type
#device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

#----------


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=16, T=1024)
torch.set_float32_matmul_precision('high')

#overiding vocab size with padded tokens to make it more even factor of 2 number
model = model.to(device)
model = GPT(GPTConfig(vocab_size=50304)) 
model = torch.compile(model)

#wandb tracking model
wandb.watch(model)
#AdamW improves upon Adam optimizer by decoupling the weight decay from the gradient update rule, by directly applying weight decay to the weights before gradient update step
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
#logits, loss = model(x, y)
#at initiazation we expect uniform probability over token(no favaourism) so {-ln(1/50157) = 10.82} close enough
for i in range(50*20):
    t0 = time.time()
    x, y = train_loader.next_batch()
    #move our tensors from cpu to device
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        #import code; code.interact(local=locals())
    loss.backward()
    optimizer.step()
    #torch.mps.synchronize()#torch.cuda.synchronize()
    device_synchronise()
    t01 = time.time()
    dt = (t01-t0)*1000#for time diff in millisecond
    dts = (t01-t0)#for time diff seconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t01-t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, {dts:.2f}sec, tec/sec: {tokens_per_sec:.2f}") # .item converts the 1 element tensor to a float and is moved to cpu
    wandb.log({"step": i, "loss": loss.item(), "dt": f"{dt:.2f}ms", "dts": f"{dts:.2f}s", "tokens_per_sec": f"{tokens_per_sec:.2f}"})



import sys; sys.exit(0)
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype= torch.long)#(8,) , 8 tokens counted
tokens = tokens.unsqueeze(0).repeat(num_returned_seq, 1) #(5,8) replicating it to 5 rows
x = tokens.to(device) # x is our idx, to get the 9th token

print("it works not crushed yet yyy")

#!Generate
#set seed to 42
torch.manual_seed(42)
#torch.cuda.manual_seed(42)
while x.size(1) < max_length: # T is less than max length
    #forwars the model to get logits
    with torch.no_grad():
        #since we specify the device type in idx while GPT forward initialization, the  tensor location wont miss match as on CPU or GPU
        logits = model(x) #(B,T,vocab size)
        #take logits at the last position
        logits = logits[:,-1,:] #(B, vocab size)....correct but inefficient sampling
        #get the probabilities
        probs = F.softmax(logits, dim=-1)
        #do top-k sampling of (50 huggingface pipeline default)
        #topk_probs here becomes(5, 50) topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim= -1)
        ix = torch.multinomial(topk_probs, 1)#(B, 1)
        #gather corrersponding indices
        xcol = torch.gather(topk_indices, -1, ix) #(B, 1)
        #append to the seq
        x= torch.cat((x, xcol), dim=1)

#print the generated text
for i in range(num_returned_seq):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print('>',decoded)



