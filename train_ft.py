import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from pathlib import Path

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