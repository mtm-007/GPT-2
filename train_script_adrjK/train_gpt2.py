from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import sys
import time
import tiktoken


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device used: ", device)

class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #key, query, value projection for all heads, but in a batch
        #one fused (for speed and convience but same number of params) big layer than 3 three separate layers for K,Q,V layers
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # a flag for this medule
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #not really a bias more of a mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1, config.block_size, config.block_size))
        
    def forward(self, x):
        B,T,C = x.size() #batch_size, sequence length, embedding dimentionality (n_embd)
        #calculate query, key, values for all heads in batch and move head forward to be the batch
        #nh is "number of heads", hs is "head size", and C "number of channels" = nh * hs
        #e.g. in GPT-2(124M), n_head=12, hs=64, so nh*hs=C=768 channels in the transformers
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        #commented for flash attention
        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        #flash attention
        y =F.scaled_dot_product_attention(q,k,v, is_causal=True)
        #transposing and reshaping after flash attention
        y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        #output projection
        #c_proj is used to project it back to the model embedding space so that the next layer can use it
        y = self.c_proj(y)
        return y
        
class MLP(nn.Module):
    """ 
    the 4 * projection scaling is used to scale for higher representation learning, and its a parameter that can be changed 4 is used based experiments papers
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # tanh approximation sued based on historcal performance in tensorflow, now?
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        """
        Andrej k. wisdom: attention acts as aggregation weighted sum function as Reduce function, the communcation part
                         MLP(FFW) acts as a map function  here the thinking stage where applied to each token individiually and think about the information gathered from attention
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) #single clear residual stream for the gradients to to branch the same backdrop gradient equally(+ backdrops equal grandients)
        x = x + self.mlp(self.ln_2(x)) #the layernorm isnot applied to the residual
        return x

@dataclass
class GPTConfig:
    """
    this config dataclass is implemented to pass parameters structurally with config instead of 
    def __init__(self, vocab_size, n_embd, n_layer, n_head, block_size) and manually pass the parameters
    """
    block_size: int = 1024
    vocab_size: int = 502304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


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
        #biggest MatMul 768 to 50257 -> 38.6M params
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight sharing/Tying scheme, it saves 38M parameters space too  
        self.transformer.wte.weight = self.lm_head.weight # the wte will copy lm head weight and be orphaned and cleaned later

        #init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """ as per openai gpt2 initialization"""
        if isinstance(module, nn.Linear):
            std =0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std +=(2* self.config.n_layer)**-0.5 #initialization for the residial layers (1/sqrt(N layers))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)#pytorch initializes bias as a uniform by default
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        #idx is of shape (B,T)
        B,T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block_size is {self.config.block_size}"
        #forward the token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long,device=idx.device)#shape (T)
        pos_emb = self.transformer.wpe(pos) #positional embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) #token embeddings of shape (B,T, n_embd)
        x = tok_emb + pos_emb
        #forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        #forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B,T, vocab_size)
        loss = None
        if targets is not None:
            #cross entropy doesnt take 3d dim, so flatten to 2d dim: inputs->(B*T, vocab_size), targets flat 1d (B*T)
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
        return logits,loss

    @classmethod
    def from_pretrained(cls, model_type):
        """ Loads pretrained GPT-2 model weights from huggingface """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  #124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), #350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), #774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), #1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 #always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        #create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        #create state_dict for both our model and for weight from huggingface
        sd = model.state_dict()
        sd_keys = sd.keys()
        # ignoring the buffer(they are not parameters) biases that come with the autoregressive mask
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #discard this mask /buffer, not a param

        #init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict() 
        
        #copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] #ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] #same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        #basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla
        #this means that we have to tranpose those weights when we import then
        assert len(sd_keys_hf)== len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                #special treatment for the Conv1D weights we need to tranpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                #vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
#----------------------------------------------
#dataloader
class Dataloaderlite:
    """ simple dataloader keeps batches in cpu"""
    def __init__(self, B,T):
        self.B = B
        self.T = T

        with open("../data/input.txt", "r")as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)// (B*T)} batches")

        #state for batching over data, here B*T at a time
        self.current_position = 0

    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        #advance the postion in the tensor
        self.current_position += B*T
        #if loading the next batch would be out of bounds, reset
        if self.current_position +(B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y

import torch
import gc

def cleanup_memory(*tensors):
    """ Deletes the provided tensors, triggers garbage collection,
        and empties CUDA cache if available. """
    for t in tensors:
        del t
    gc.collect()
    
    # Only if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
"""
gradient clipping, what actually happens
total_norm = sqrt(sum(p.grad.norm(2)**2 for p in model.parameters()))
if total_norm > max_norm:
    scale = max_norm / (total_norm + 1e-6)
    for p in model.parameters():
        p.grad.mul_(scale)
return total_norm
"""
#----------------------------------------------

train_loader = Dataloaderlite(B=4,T=1024)
#set to tf32 when available, only available in GPU ampere feature
torch.set_float32_matmul_precision("high")
#model = GPT.from_pretrained('gpt2')
#with out using pretrained weights

#get logits
model = GPT(GPTConfig(vocab_size=50304)) #vocab_size use better 8,16,32 divisble number
model.to(device)
model= torch.compile(model)

#learning rate scheduler
max_lr = 6e-4
min_lr = max_lr*0.1
warm_up_steps = 10
max_steps = 84

def get_lr(it):
    #1. linear warmup for warmup_iters steps
    if it < warm_up_steps:
        return max_lr * (it+1)/warm_up_steps
    #2. if lr > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    #3. in between, use cosine decay to min learning rate
    decay_ratio = (it - warm_up_steps)/ (max_steps - warm_up_steps)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

#optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(max_steps):#iterations
    t0=time.time()
    x,y = train_loader.next_batch()
    x,y = x.to(device), y.to(device) #move the batches from cpu to device
    optimizer.zero_grad()
    #use pytorch autocast(automatic mixed precision) for model and loss only leave others 
    #the logits activations changes to bf16 but the model weight parameters stay at ft32
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x,y)
        #import code;code.interact(local=locals()) #inline python shell, manual debugger
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #determine and set the learning rate for the iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    #torch.cuda.synchronize()
    t1=time.time()
    dt= (t1-t0) #in diff in seconds
    tokens_processed = train_loader.B*train_loader.T
    tokens_per_sec = tokens_processed/ dt
    print(f"step {step:4d}, loss: {loss.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt*1000:.2f}ms, token/sec throughput: {tokens_per_sec:.2f}") #calling .item() here ships the float to cpu, if gpu loss will be in gpu then .item() makes a copy to cpu to print

    #call garbage collector
    cleanup_memory(x,y,loss)

#sanity check loss should be -ln(1/50257)roughly 10.8
sys.exit(0) #to skip sampling logic here

model.eval()
num_return_sequence = 5
max_length =60
#prefix tokens
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, Andrej Kharpathy is the King, best teacher for deep learning")
tokens = torch.tensor(tokens, dtype=torch.long) #(16,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequence,1) # (5, 16)
x = tokens.to(device)

#genetate right now x is (B,T) where B=5, T= 16
#set seed to 42
torch.manual_seed(1337)
if torch.cuda.is_available(): torch.cuda.manual_seed(1337)

while x.size(1) < max_length:
    #forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        #take the logits at the last position 
        # only take the last column logits as indices are added one column per time for all rows(5,here), inefficient sampling 
        logits = logits[:,-1,:] #(B,vocab_size)
        #get the probabilities
        probs = F.softmax(logits, dim=-1)
        #do top-k sampling for 50 (huggingface pipeline default)
        #topk_probs here becomes (5, 50), topk_indices is(5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        #select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs,1) #(B,1)
        #gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) #(B, 1)
        #append to the sequence 
        x = torch.cat((x, xcol), dim=1)

#print the generated text
for i in range(num_return_sequence):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)