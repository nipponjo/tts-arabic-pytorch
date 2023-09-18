# %%
import argparse
import os
import torch
from torch.utils.data import DataLoader
from models.tacotron2.tacotron2_ms import Tacotron2MS

from utils import get_config
from utils.data import ArabDataset, text_mel_collate_fn
from utils.logging import TBLogger
from utils.training import batch_to_device, save_states

from models.tacotron2.loss import Tacotron2Loss

# %%

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default="configs/nawar_tc2.yaml", help="Path to yaml config file")
    args = parser.parse_args()
    config_path = args.config
except:
    config_path = './configs/nawar_tc2.yaml'

# %%

config = get_config(config_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set random seed
if config.random_seed != False:
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    import numpy as np
    np.random.seed(config.random_seed)

# make checkpoint folder if nonexistent
if not os.path.isdir(config.checkpoint_dir):
    os.makedirs(os.path.abspath(config.checkpoint_dir))
    print(f"Created checkpoint_dir folder: {config.checkpoint_dir}")

# datasets
train_dataset = ArabDataset(txtpath=config.train_labels, 
                            wavpath=config.train_wavs_path,
                            label_pattern=config.label_pattern)
# test_dataset = ArabDataset(config.test_labels, config.test_wavs_path)

# optional: balanced sampling
sampler, shuffle, drop_last = None, True, True
if config.balanced_sampling:
    weights = torch.load(config.sampler_weights_file)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights, len(weights), replacement=False)
    shuffle, drop_last = False, False

# dataloaders
train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          collate_fn=text_mel_collate_fn,
                          shuffle=shuffle, drop_last=drop_last,
                          sampler=sampler)

# test_loader = DataLoader(test_dataset,
#                          batch_size=config.batch_size, drop_last=False,
#                          shuffle=False, collate_fn=text_mel_collate_fn)

# %% Generator
model = Tacotron2MS(n_symbol=40, num_speakers=40)
model = model.to(device)
model.decoder.decoder_max_step = config.decoder_max_step

optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=config.g_lr, 
                              betas=(config.g_beta1, config.g_beta2), 
                              weight_decay=config.weight_decay)
criterion = Tacotron2Loss(mel_loss_scale=1.0)

# %%
# resume from existing checkpoint
n_epoch, n_iter = 0, 0

if config.restore_model != '':
    state_dicts = torch.load(config.restore_model)
    model.load_state_dict(state_dicts['model'])
    if 'optim' in state_dicts:
        optimizer.load_state_dict(state_dicts['optim'])
    if 'epoch' in state_dicts:
        n_epoch = state_dicts['epoch']
    if 'iter' in state_dicts:
        n_iter = state_dicts['iter']

# %%
# tensorboard writer
writer = TBLogger(config.log_dir)

# %%

def trunc_batch(batch, N):
    return (batch[0][:N], batch[1][:N], batch[2][:N],
            batch[3][:N], batch[4][:N])

# %% TRAINING LOOP

model.train()

for epoch in range(n_epoch, config.epochs):
    print(f"Epoch: {epoch}")
    for batch in train_loader:

        if batch[-1][0] > 2000:
            batch = trunc_batch(batch, 6)

        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch_to_device(batch, device)

        y_pred = model(text_padded, input_lengths,
                       mel_padded, output_lengths,
                       torch.zeros_like(output_lengths))
        mel_out, mel_out_postnet, gate_out, alignments = y_pred
       
        # GENERATOR
        loss, meta = criterion(mel_out, mel_out_postnet, mel_padded,
                               gate_out, gate_padded)  

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip_thresh)
        optimizer.step()

        # LOGGING
        meta['loss'] = loss.clone().detach()

        print(f"loss: {loss.item()}, grad_norm: {grad_norm.item()}")

        writer.add_training_data(meta, grad_norm.item(),
                                 config.learning_rate, n_iter)


        if n_iter % config.n_save_states_iter == 0:
            save_states(f'states.pth', model, 
                        optimizer, n_iter, 
                        epoch, None, config)

        if n_iter % config.n_save_backup_iter == 0 and n_iter > 0:
            save_states(f'states_{n_iter}.pth', model, 
                        optimizer, n_iter, 
                        epoch, None, config)

        n_iter += 1

    # VALIDATE
    # val_loss = validate(model, test_loader, writer, device, n_iter)
    # print(f"Validation loss: {val_loss}")


save_states(f'states.pth', model,
            optimizer, n_iter, 
            epoch, None, config)


# %%
