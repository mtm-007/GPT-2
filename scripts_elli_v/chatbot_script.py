import torch, sys
import torch.nn as nn
import wandb
from torch.nn import functional as F
from Transformer_functions import GPTLanguagemodel

wandb.login()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.manual_seed(1337)

block_size = 512
batch_size = 48
max_iters = 1000
learning_rate = 3e-4
eval_iters = 100
eval_interval = 100
n_embed = 384
n_layer = 4
n_head = 4
dropout = 0.2

# wandb tracking initialization
wandb.init(project="text-generation", name="interactive_generation",
      config={
            "block_size" : 512,
            "batch_size" :48,
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

model = GPTLanguagemodel(vocab_size).to(device)

#print('loading model parameters..')
# with open('models/model-01.pkl', 'rb') as f:
#     model = pickle.load(f)
# print('loading model succesfully..')

#add torch compile 
m = torch.compile(model)
wandb.watch(m)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

try:
    while True:
        prompt = input("prompt:\n")
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
        generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=200)[0].tolist())
        print(f"completion:\n{generated_chars}")
        
        wandb.log({"prompt": prompt, "completion": generated_chars})

except KeyboardInterrupt:
    print("\nExiting gracefully...")
