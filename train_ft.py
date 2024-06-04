import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
import warnings

from dataset_t import BilingualDataset, casual_mask
from model_2 import build_transformer
from config_ft import get_config, get_weights_file_path

def get_all_sentence(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
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
    train_ds_size = int(0.9 * len(ds_raw))
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
        max_len_trg = max(max_len_trg, len(trg_ids))

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
    #device= 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"training with {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, valid_dataloader, tokenizer_src, tokenizer_trg = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size()).to(device)
    

    #Tensorboard
    writer = SummaryWriter(config['experiment'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    #setting up checkpoints if training crushes
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model weights {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_state']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # Batch, seq_len
            decoder_input = batch['decoder_input'].to(device) # Batch, seq_len
            encoder_mask = batch['encoder_mask'].to(device) # (Batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (Batch, 1, seq_len, seq_len)
        
        #Run the tensors through the Transformer 
        encoder_output = model.encode(encoder_input, encoder_mask)#(batch, seq_len, d_model)
        decoder_output = model.encode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
        proj_out = model.projection(decoder_output) #(batch, seq_len, trg_vocab_size)

        label = batch['label_output'].to(device) #(batch, seq_len)

        #(batch, seq_len, trg_vocab_size) ->(batch * seq_len, trg_vocab_size)
        loss = loss_fn(proj_out.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1))
        batch_iterator.set_postfix({f"loss": f"{loss.item(): 6.3f}"})

        #log the loss to the Tensorboard
        writer.add_scalar('train_loss', loss.item(), global_step)
        writer.flush()

        #backdropagate the loss
        loss.backward()

        #update the loss or weights
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        

    #save the model at the end of every epoch
    model_filename = get_weights_file_path(config, f'{epoch:02d}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'opitimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }, model_filename)

if __name__ == '__main__':
    #warnings.fiterwarnings('ignore')
    config = get_config()
    train_model(config)


