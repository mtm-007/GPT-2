import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from pathlib import Path

from dataset_ft import BilingualDataset, casual_mask
from model_2 import build_transformer

def get_all_sentence(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNk]'))
        tokenizer.pre_tokenizer = Whitespace()
        #the speacial tokens are unknown, paddding, start of seq, end of seq
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentence(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    #configured to work for different languages
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_trg"]}', split= 'train')

    #build tokenizer
    tokenizer_src = get_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_trg = get_build_tokenizer(config, ds_raw, config["lang_trg"])

    #split for train to 90% and 10% for validation
    train_ds_size = len(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) 

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_trg, config["lang_src"], config["lang_trg"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_trg, config["lang_src"], config["lang_trg"], config["seq_len"])

    max_len_src = 0
    max_len_trg = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids 
        trg_ids = tokenizer_src.encode(item['translation'][config['lang_trg']]).ids 
        max_len_src = max(max_len_src, len(src_ids))
        max_len_trg = max(max(max_len_trg, len(trg_ids)))

    print(f"Max length of the source sentence {max_len_src}")
    print(f"Max length of the target sentence {max_len_trg}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, valid_dataloader, tokenizer_src, tokenizer_trg

def get_model(config, src_vocab_size, trgt_vocab_size):
    #for limited GPU resource reduce the n_head, number of layers
    model = build_transformer(src_vocab_size, trgt_vocab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    #Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"training with {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    