
import math
import sys,os
import time
import tiktoken
import inspect
import gc
import wandb
import numpy as np
from dataclasses import dataclass
from dataclasses import asdict
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag_evals import render_example, iterate_examples

#-----wandb logging-------
wandb.init(
        project='nano-gpt-tracking-test',
        name=f"run_{int(time.time())}",  # unique run name
)
#----------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device used: ", device)
wandb.log({"device": device})

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
    vocab_size: int = 50304
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
        wandb.log({"loading_weight_from_pretrained_gpt": model_type})

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
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        #start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        #create optioin groups. any parameters that is 2D will be decayed, otherwise no
        #i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms dont.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [ p for n,p in param_dict.items() if p.dim() <2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        wandb.log({
        "num_decay_params": num_decay_params,
        "num_nodecay_params": num_nodecay_params,
        "num_decay_tensors": len(decay_params),
        "num_nodecay_tensors": len(nodecay_params)
        })

        #create AdamW optimizer and use fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        wandb.log({"use_fused_adamw": use_fused})
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
#----------------------------------------------
#dataloader
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # per kharpathy Readme PR update pytorch dont accept uint16 arrays directly.
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class Dataloaderlite:
    """ simple dataloader keeps batches in cpu"""
    def __init__(self, B,T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        #get the shard filename
        data_root = "edu_fineweb1B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for the split {split}")
            wandb.log({"num_shards": len(shards), "shards_for_the_split": split})
        self.reset()
        
    def reset(self):
        #state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        #advance the postion in the tensor
        self.current_position += B*T*self.num_processes
        #if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B*T*self.num_processes+1) > len(self.tokens):
            self.current_shard = (self.current_shard +1) %len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

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
#-------------------------hellaSwag eval helper function--------

def get_most_likely_row(tokens, mask, logits):
    #evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1,:]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    #now get the average loss just for the completion region (where mask ==1), in each row 
    shift_mask = (mask[..., 1:]).contiguous() #we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    #sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    #now we have a loss for each of the 4 completions
    #the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm 

#------------------------------------------------------------------

#---------Distributed training------
#run the training loop
#DDP launch for multi GPU e.g. 8
#torchrun --standalone --nproc_per_node=8 train_gpt2.py

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.distributed as dist
#set up DDP (distributed data parallel)
#torchrun command sets the new env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 #is this a ddp run
if ddp:
    #use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank ==0 #this process will do logging, checkpointing etc.
    
else:
    #vanila, non-DDP run
    ddp_rank=0
    ddp_local_rank=0
    ddp_world_size=1
    master_process=True
    #auto detect device
    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    wandb.log({"device": device})

#pytorch issue to lookup for with autocast device changed to device_type
device_type = "cuda" if device.startswith("cuda") else "cpu"
#----------------------------------------------
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 32768 #524288 #2**19 #~0.5M tokens
B = 4
T = 1024
assert total_batch_size % (B *T * ddp_world_size) ==0, "make sure its divisible by B*T"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    wandb.log({
        "total_desired_batch_size": total_batch_size,
        "calculated_grad_accum_steps": grad_accum_steps
    })

# print("I am GPU/CPU:", ddp_rank)
# print("test success!")
# import sys; sys.exit(0)

train_dataloader = Dataloaderlite(B=B,T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_dataloader = Dataloaderlite(B=B,T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

#set to tf32 when available, only available in GPU ampere feature
torch.set_float32_matmul_precision("high")
#model = GPT.from_pretrained('gpt2')
#with out using pretrained weights

#create model
config = GPTConfig() #vocab_size use better 8,16,32 divisble number
model = GPT(config)
model.to(device)

wandb.config.update(asdict(config)) # logs block_size, vocab_size, n_layer, etc.

use_compile = False # torch.compile interfaces with hellaSwag eval and generation.
if use_compile:
    model= torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model #always contain the raw unwrapped model

wandb.watch(raw_model)

num_of_parameters = sum(p.numel() for p in raw_model.parameters())/1e6
print(f"{num_of_parameters:.2f} M parameters")
wandb.log({"num_of_parameters": num_of_parameters})
#learning rate scheduler
max_lr = 6e-4
min_lr = max_lr*0.1
warm_up_steps = 715
max_steps = 50

# Add additional training hyperparameters not in GPTConfig
wandb.config.update({
    "device": device,
    "B": B,
    "T": T,
    "max_lr": max_lr,
    "min_lr": min_lr,
    "warm_up_steps": warm_up_steps,
    "max_steps": max_steps,
    "use_compile": use_compile,
})


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
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

#create the log directory we will write checkpoints to log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:#open for writing to clear the file
    pass

for step in range(max_steps):#iterations
    t0=time.time()
    last_step = (step == max_steps -1)

    #once in a while evaluator our evaluation loss
    if step %2500 ==0 or last_step:
        model.eval()
        val_dataloader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x,y = val_dataloader.next_batch()
                x,y = x.to(device),y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x,y)
                loss = loss / val_loss_steps
                val_loss_accum +=loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            wandb.log({"val_loss": val_loss_accum.item()})
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                wandb.log({
                "step": step,
                "val": val_loss_accum.item()
                })
            if step > 0 and (step % 5000 == 0 or last_step):
                #optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(), #add optimizer state for later resuming training
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'rng_state': torch.get_rng_state(), #get random seed from pytorch for resuming training where it left off
                    'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                }
                #you might also want to add optimizer.state() and
                #rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

                # log checkpoint to W&B
                artifact = wandb.Artifact(f"model_step_{step:05d}", type="model")
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
                
    #once in a while evalute hellaSwag
    if (step %250 ==0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i,example in enumerate(iterate_examples("val")):
            #only process example where i %ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            #render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            #get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total +=1
            num_correct_norm += int(pred_norm == label)
        #reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/ {num_total}= {acc_norm:.4f}")
            wandb.log({"hellaswag_acc": acc_norm})
            with open(log_file, "a")as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
            wandb.log({
            "step": step,
            "hella": acc_norm
            })
    
    #once in a while generate from the model (except step 0, which is noise)
    #disabled because torch.compile throws a scary error to be solved as Andrej Kharpathy
    #if you disable torch.compile, this code works fine
    if ((step >0 and step %250 ==0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequence = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a Andrej Kharpathy is my king language model,")
        tokens = torch.tensor(tokens, dtype=torch.long) #(16,)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequence,1) # (5, 16)
        xgen = tokens.to(device)
        #create a seperate object generator for the random number to be outside the training loop and not intefer with the global training seed
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank) #every rank is a different random seed
        while xgen.size(1) < max_length:
            #forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) #(B, T, vocab_size)
                #take the logits at the last position
                logits = logits[:,-1, :] #(B, vocab_size)
                #get the probabilities
                probs = F.softmax(logits, dim=-1)
                #do top-k sampling for 50 (huggingface pipeline default)
                #topk_probs here becomes (5, 50), topk_indices is(5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                #select a token from the top-k probabilities
                #note multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs,1, generator=sample_rng) #(B,1)
                #gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) #(B, 1)
                #append to the sequence 
                xgen = torch.cat((xgen, xcol), dim=1)
            
            # print the generated text
            table = wandb.Table(columns=["step", "rank", "sample_index", "text"])
            for i in range(num_return_sequence):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")
                table.add_data(step, ddp_rank, i, decoded)
            wandb.log({"generated_samples": table})

    # ===== LOAD LATEST CHECKPOINT IF AVAILABLE =====
    checkpoint_files = glob.glob(os.path.join(log_dir, "model_*.pt"))

    if checkpoint_files:
        # Sort by step number extracted from filename: model_00005.pt → 5
        checkpoint_files = sorted(
            checkpoint_files,
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )
        latest_ckpt = checkpoint_files[-1]  # highest step checkpoint

        # Load checkpoint
        checkpoint = torch.load(latest_ckpt, map_location=device)
        raw_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Restore RNG states for reproducibility
        torch.set_rng_state(checkpoint['rng_state'])
        if torch.cuda.is_available() and checkpoint.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        
        start_step = checkpoint['step'] + 1
        print(f"Resuming training from checkpoint {latest_ckpt} at step {start_step}")
        wandb.log({"Resuming_training_from_checkpoint": latest_ckpt, "step": start_step})
    else:
        # No checkpoint found → start from scratch
        start_step = 0
        torch.manual_seed(1337)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)
        print("No checkpoint found. Starting training from scratch.")
        wandb.log({"check_stats":"No checkpoint found. Starting training from scratch."})
    # ------------------------------------

    #training loop
    #do one step of optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step_batch in range(grad_accum_steps):
        x,y = train_dataloader.next_batch()
        x,y = x.to(device), y.to(device) #move the batches from cpu to device
        if ddp:
            model.require_backward_grad_sync = (micro_step_batch == grad_accum_steps -1)
        #use pytorch autocast(automatic mixed precision) for model and loss only leave others 
        #the logits activations changes to bf16 but the model weight parameters stay at ft32
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x,y)
            #import code;code.interact(local=locals()) #inline python shell, manual debugger
        #we have to scale the loss down to account for gradient accumulation,
        #because the gradients just add on each successive backwards().
        #addition of the gradients corresponds to a SUM in the objective, but
        #instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss/grad_accum_steps #scale down to cover the sum over the gradient accum stage
        loss_accum += loss.detach()
        loss.backward()
    if ddp: 
        dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #determine and set the learning rate for the iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type=="cuda":
        torch.cuda.synchronize()
    t1=time.time()
    dt= (t1-t0) #in diff in seconds
    tokens_processed = train_dataloader.B * train_dataloader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed/ dt
    if master_process:
        print(f"step {step:4d}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt*1000:.2f}ms, token/sec throughput: {tokens_per_sec:.2f}") #calling .item() here ships the float to cpu, if gpu loss will be in gpu then .item() makes a copy to cpu to print
        wandb.log({
        "train_loss": loss_accum.item(),
        "learning_rate": lr,
        "grad_norm": norm,
        "tokens_per_sec": tokens_per_sec,
        "step": step
    })
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
        
        # Log to W&B as usual
        wandb.log({
            "step": step,
            "train_loss": loss_accum.item()
        })
    #call garbage collector
    cleanup_memory(x,y,loss)

# training loop ends and add log_file to wandb
if master_process:
    artifact = wandb.Artifact("final_training_logs", type="log")
    artifact.add_file(log_file)
    wandb.log_artifact(artifact)

if ddp:
    destroy_process_group()
    