import tiktoken
import torch

class DataLoaderLite():
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('..data/input.txt', 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.token = torch.tensor(tokens)
        print(f' loaded {len(tokens)} tokens')
        print(f"1 epoch = {len(tokens) // (B * T)} batches")

    #state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B*T+1 ]
        x = (buf[:-1]).view(B,T)#inputs
        y = (buf[1:]).view(B,T)#targets
        #advance each batch by (B*T)
        self.current_position += B*T
        #checking if loading next batch is out of bound
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0 #return to the new next epoch
        return x, y