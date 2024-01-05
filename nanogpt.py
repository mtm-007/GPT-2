
import torch, math
import torch.nn as nn
from torch.nn import functional as F


#hyperparameters
batch_size= 32
block_size = 8
max_iters = 3000
eval_iterval = 300
learning_rate = 1e-2
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#-------
torch.manual_seed(1337)


# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder takes a string and outputs list of intergers
decoder = lambda l: ''.join([itos[i] for i in l]) # decoder takes intergers and outputs a strings

data = torch.tensor(encode(text), dtype = torch.long)
# lets split our dataset in to train and validatioin set
n = int(0.9*len(text)) # the first 90% will be for training and 10% for validation
train_data = data[:n]
val_data = data[n:]

#dataloading
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx, targets=None): 
        #idx, and tagets are both in dimention (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # B,T, C, Batch, T (block_size), C(channel)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # pytorch expects the dimentions to be in B, C, T so lets reshape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)# we can also use -1 and pytorch will sort out but to be explicit
            loss = F.cross_entropy(logits, targets)
        return logits,loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits,loss = self(idx)
            #focus on the last of time step
            logits = logits[:,-1,:] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            id_next = torch.multinomial(probs, num_samples=1)# (B, C)
            idx = torch.cat((idx, id_next), dim = 1) # (B, T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)

for iter in range(max_iters): 
    if iter % eval_iterval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb,yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device= device)
print(decoder(m.generate(context, max_tokens=500)[0].tolist()))


