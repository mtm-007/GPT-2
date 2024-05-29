import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_trg, src_lang, trg_lang, seq_len ) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.src_lang = src_lang
        self.trg_lang = trg_lang

        #get tensor id for the special tokens
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype= torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype= torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype= torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        trg_text = src_target_pair['translation'][self.src_lang]

        enc_input_tokens = self.tokenizer_src.encdoe(src_text).ids
        dec_input_tokens = self.tokenizer_trg.encode(trg_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        '''The decoder input generally starts with a [SOS] token and may not require 
        an additional end token since the decoder operates in an auto-regressive manner
        (generating one token at a time until an end-of-sequence token is generated).'''
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
    
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            #the padding shouldnt be negative
            raise ValueError('Sentence is too long') 