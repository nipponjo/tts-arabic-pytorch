import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.fastpitch import net_config
from models.fastpitch.fastpitch.model import FastPitch
from models.fastpitch.fastpitch.data_function import (TTSCollate, batch_to_gpu)
from models.fastpitch.fastpitch.loss_function import FastPitchLoss
from models.fastpitch.fastpitch.attn_loss_function import AttentionBinarizationLoss
from utils.data import DynBatchDataset
from utils import get_config
from utils.training import save_states
# %%

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default="configs/nawar_fp.yaml", help="Path to yaml config file")
    args = parser.parse_args()
    config_path = args.config
except:
    config_path = './configs/nawar_fp.yaml'

# %%

config = get_config(config_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make checkpoint folder if nonexistent
if not os.path.isdir(config.checkpoint_dir):
    os.makedirs(os.path.abspath(config.checkpoint_dir))
    print(f"Created checkpoint folder @ {config.checkpoint_dir}")


train_dataset = DynBatchDataset(
    txtpath=config.train_labels,
    wavpath=config.train_wavs_path,
    label_pattern=config.label_pattern,
    f0_dict_path=config.f0_dict_path,
    f0_mean=config.f0_mean, f0_std=config.f0_std,
    max_lengths=config.max_lengths,
    batch_sizes=config.batch_sizes,
    )

# %%

collate_fn = TTSCollate()

config.batch_size = 1
sampler, shuffle, drop_last = None, True, True
train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          collate_fn=lambda x: collate_fn(x[0]),
                          shuffle=shuffle, drop_last=drop_last,
                          sampler=sampler)

# %% Generator

model = FastPitch(**net_config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=config.g_lr, 
                              betas=(config.g_beta1, config.g_beta2), 
                              weight_decay=config.weight_decay)

criterion = FastPitchLoss()
attention_kl_loss = AttentionBinarizationLoss()

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
else:
    # from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/fastpitch__pyt_ckpt
    model_sd = torch.load('G:/models/fastpitch/nvidia_fastpitch_210824+cfg.pt')
    model.load_state_dict(
        {k.removeprefix('module.'): v for k, v in model_sd['state_dict'].items()})

# %%
writer = SummaryWriter(config.log_dir)

# %% TRAINING LOOP

model.train()

for epoch in range(n_epoch, config.epochs):
    train_dataset.shuffle()
    for batch in train_loader:

        x, y, _ = batch_to_gpu(batch)

        y_pred = model(x)

        mel_out, *_, attn_soft, attn_hard, _, _ = y_pred
        _, _, mel_padded, output_lengths, *_ = x       

        # generator step
        loss, meta = criterion(y_pred, y)  
    
        binarization_loss = attention_kl_loss(attn_hard, attn_soft)        
        loss += 1.0 * binarization_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1000.)
        optimizer.step()

        # LOGGING
        meta['kl_loss'] = binarization_loss.clone().detach()

        print(f"loss: {meta['loss'].item()} gnorm: {grad_norm}")

        for k, v in meta.items():
            writer.add_scalar(f'train/{k}', v.item(), n_iter)

        if n_iter % config.n_save_states_iter == 0:
            save_states(f'states.pth', model, 
                        optimizer, n_iter, 
                        epoch, net_config, config)

        if n_iter % config.n_save_backup_iter == 0 and n_iter > 0:
            save_states(f'states_{n_iter}.pth', model, 
                        optimizer, n_iter, 
                        epoch, net_config, config)

        n_iter += 1


save_states(f'states.pth', model, 
            optimizer, n_iter,
            epoch, net_config, config)


# %%

# (mel_out, 0
# dec_mask, 1
# dur_pred, 2
# log_dur_pred, 3
# pitch_pred, 4
# pitch_tgt, 5
# energy_pred, 6
# energy_tgt, 7
# attn_soft, 8
# attn_hard, 9
# attn_dur, 10
# attn_logprob, 11
# ) = model_out

# x = [text_padded, input_lengths, mel_padded, output_lengths,
#         pitch_padded, energy_padded, speaker, attn_prior, audiopaths]

# y = [mel_padded, input_lengths, output_lengths]
