# %%
import os
import torch
from torch.utils.data import DataLoader
from models.tacotron2.tacotron2_ms import Tacotron2MS

from utils import get_config
from utils.data import ArabDataset, text_mel_collate_fn
from utils.logging import TBLogger
from utils.training import *

from models.common.loss import Discriminator3, extract_chunks, calc_fmatch_loss
from models.tacotron2.loss import Tacotron2Loss

# %%
def save_states(fname, model, model_d, optimizer, optimizer_d, 
                n_iter, epoch, config):
    torch.save({'model': model.state_dict(),
                'model_d': model_d.state_dict(),                
                'optim': optimizer.state_dict(),
                'optim_d': optimizer_d.state_dict(),
                'epoch': epoch, 'iter': n_iter,       
                },
               f'{config.checkpoint_dir}/{fname}')

# %%

config_path = './configs/nawar.yaml'

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
if config.cache_dataset:
    print('Caching datasets ...')
train_dataset = ArabDataset(config.train_labels, config.train_wavs_path,
                            cache=config.cache_dataset)
test_dataset = ArabDataset(config.test_labels, config.test_wavs_path,
                           cache=config.cache_dataset)

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

test_loader = DataLoader(test_dataset,
                         batch_size=config.batch_size, drop_last=False,
                         shuffle=False, collate_fn=text_mel_collate_fn)

# construct model
model = Tacotron2MS(n_symbol=40)
model = model.to(device)
model.decoder.decoder_max_step = config.decoder_max_step

# optimizer
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-4, betas=(0.0, 0.99),
                              weight_decay=config.weight_decay)


tar_len, Nchunks = 128, 8
critic = Discriminator3(1, 32).to(device)

optimizer_d = torch.optim.AdamW(critic.parameters(),
                                lr=1e-4, betas=(0.0, 0.99),
                                weight_decay=config.weight_decay)


criterion = Tacotron2Loss(mel_loss_scale=1.0, 
                          attn_loss_scale=0.1)


# %%

# config.restore_model = './checkpoints/exp_tc2_adv/states.pth'

# resume from existing checkpoint
n_epoch, n_iter = 0, 0
if config.restore_model != '':
    state_dicts = torch.load(config.restore_model)
    model.load_state_dict(state_dicts['model'])
    if 'model_d' in state_dicts:
        critic.load_state_dict(state_dicts['model_d'], strict=False)
    if 'optim' in state_dicts:
        optimizer.load_state_dict(state_dicts['optim'])
    if 'optim_d' in state_dicts:
        optimizer_d.load_state_dict(state_dicts['optim_d'])
    if 'epoch' in state_dicts:
        n_epoch = state_dicts['epoch']
    if 'iter' in state_dicts:
        n_iter = state_dicts['iter']

# %%
# tensorboard writer
# config.log_dir = 'logs/exp_tc2_adv2'
writer = TBLogger(config.log_dir)

# %%

def trunc_batch(batch, N):
    return (batch[0][:N], batch[1][:N], batch[2][:N],
            batch[3][:N], batch[4][:N])

# %%

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

        # extract chunks for critic
        tar_len_ = min(output_lengths.min().item(), tar_len)
        mel_ids = torch.randint(0, mel_out_postnet.size(0), (Nchunks,)).to(device)
        ofx_perc = torch.rand(Nchunks).to(device)
        out_lens = output_lengths[mel_ids]

        ofx = (ofx_perc * (out_lens + tar_len_) - tar_len_/2) \
            .clamp(out_lens*0, out_lens - tar_len_).long()

        chunks_org = extract_chunks(
            mel_padded, mel_ids, ofx, tar_len_)  # mel_padded: B F T
        chunks_gen = extract_chunks(
            mel_out_postnet, mel_ids, ofx, tar_len_)  # mel_out_postnet: B F T

        chunks_org_ = (chunks_org.unsqueeze(1) + 4.5) / 2.5
        chunks_gen_ = (chunks_gen.unsqueeze(1) + 4.5) / 2.5

        # DISCRIMINATOR
        d_org, fmaps_org = critic(chunks_org_.requires_grad_(True))
        d_gen, _ = critic(chunks_gen_.detach())

        loss_d = 0.5*(d_org - 1).square().mean() + 0.5*(d_gen).square().mean()

        critic.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # GENERATOR
        loss, meta = criterion(
            mel_out,  mel_out_postnet,
            gate_out,  mel_padded, alignments, 
            input_lengths,  output_lengths)


        d_gen2, fmaps_gen = critic(chunks_gen_)
        loss_score = (d_gen2 - 1).square().mean()
        loss_fmatch = calc_fmatch_loss(fmaps_gen, fmaps_org)
        meta['score'] = loss_score.clone().detach()
        meta['fmatch'] = loss_fmatch.clone().detach()

        loss += 5.0 * loss_score
        loss += 1.0 * loss_fmatch
        meta['loss'] = loss.clone().detach()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip_thresh)
        optimizer.step()

        # LOGGING
        print(f"loss: {loss.item()}, grad_norm: {grad_norm.item()}")

        writer.add_training_data(meta, grad_norm.item(),
                                 config.learning_rate, n_iter)


        if n_iter % config.n_save_states_iter == 0:
            save_states(f'states.pth', model, critic,
                        optimizer, optimizer_d, n_iter, 
                        epoch, config)

        if n_iter % config.n_save_backup_iter == 0 and n_iter > 0:
            save_states(f'states_{n_iter}.pth', model, critic,
                        optimizer, optimizer_d, n_iter, 
                        epoch, config)

        n_iter += 1

    # VALIDATE
    # val_loss = validate(model, test_loader, writer, device, n_iter)
    # print(f"Validation loss: {val_loss}")


save_states(f'states.pth', model, critic,
            optimizer, optimizer_d, n_iter, 
            epoch, config)


# %%

import sounddevice as sd
import matplotlib.pyplot as plt
from vocoder import load_hifigan
from vocoder.hifigan.denoiser import Denoiser

vocoder = load_hifigan(config.vocoder_state_path, config.vocoder_config_path)
vocoder = vocoder.to(device)
denoiser = Denoiser(vocoder)

test_dataset = ArabDataset(config.test_labels, config.test_wavs_path,
                           cache=config.cache_dataset)


# %%

idx = 0
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.imshow(mel_out[idx].detach().cpu(), origin='lower', aspect='auto')
ax2.imshow(mel_padded[idx].cpu(), origin='lower', aspect='auto')


# %%

model.eval()

(phonemes, mel_log) = test_dataset[1]

with torch.inference_mode():
    (mel_out,
     mel_specgram_lengths,
     alignments) = model.infer(phonemes[None, :].to(device))

    wave = vocoder(mel_out[0])
    wave_enhan = denoiser(wave, 0.005)

sd.play(0.3*wave[0].cpu(), 22050)
# sd.play(0.3*wave_enhan[0].cpu(), 22050)

plt.imshow(mel_out[0].cpu(), aspect='auto', origin='lower')

