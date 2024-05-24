import torch
import torch.nn as nn
import math 

#d_model here is the embedding vector dimention size

class InputEmbeddings(nn.Module):
    
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        

    def forward(self, x):
        # according to the paper, we need to multiply by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
        #return self.pos_encoder(self.embedding(x))

class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        #create a tensor of shape (seq_len,1)....unsqueeze(1) is used to add a dimension at the end
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        #apply the sin to even indexed dimensions and cos to odd indexed dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        #add the batch dimention for pe as its (seq_len, d_model) above
        pe = pe.unsqueeze(0) # as unit dimention in the first -> (1, seq_len, d_model)

        # the buffer is something we add with the file or state of the model but not as a learned parameter
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #multiplies 
        self.bias = nn.Parameter(torch.zeros(1)) # additive

    def forward(self, x):
        mean = x.mean(dim =-1, keepdim=True)# applied to the last dimention
        std = x.std(dim =-1, keepdim= True)
        return self.alpha * (x - mean)/(std + self.eps) + self.bias # apply the formula
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.Linear_1 = nn.Linear(d_model, d_ff) # W1 and B1 . bias ia True by default 
        self.dropout = nn.Dropout(dropout)
        self.Linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # we have input sent (batch, seq_len, d_model) -L1-> (batch, seq_len, d_ff) -L2-> (batch, seq_len, d_model)
        return self.Linear_2(self.dropout(torch.relu(self.Linear_1(x)))) # applying relu Here fused
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert d_model % n_head ==0, 'd_model is not divisible by n_head'

        self.d_k = d_model // n_head  # 
        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) # wk
        self.w_v = nn.Linear(d_model, d_model) # wv

        self.wo = nn.Linear(d_model, d_model) #wo, here first d_model is n_head* d_v. dv is after multiplied after softmax

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Module):
        d_k = query.shape[-1]

        #(batch, n_head, seq_len, d_k) --> (batch, n_head, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # transpose the last 2 dimentions (batch, seq_len, d_k) -> (batch, d_k, seq_len)
        if mask is not None:
            attention_scores.masked_fill(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1 ) # (batch, n_head, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)   # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        
        #(batch, seq_len, d_model) --> (batch, seq_len, n_head, d_k) --transpose--> (batch, n_head, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.n_head, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_head, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        #(batch, n_head, seq_len, d_k) --transpose--> (batch, seq_len, n_head, d_k) --> (batch, seq_len, d_model)
        #contigious is applied here to insure they are in continous memory after applying tranpose
        x = x.transpose(1,2).contigious().view(x.shape[0], -1, self.n_head * self.d_k)

        #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):

    def __init__(self, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # here the norm is applied before the subplayer but in the paper norm is applied later
        return x + self.dropout(sublayer(self.norm(x))) 
    

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, enc_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, enc_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class Decoder(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, enc_mask, dec_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, dec_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, enc_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, enc_mask, dec_mask):
        for layers in self.layers:
            x = layers(x, encoder_output, enc_mask, dec_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
