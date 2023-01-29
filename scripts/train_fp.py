
import os

import matplotlib.pyplot as plt
import numpy as np
import text
import torch
import torchaudio
from models.fastpitch import net_config
from models.fastpitch.fastpitch.attn_loss_function import \
    AttentionBinarizationLoss
from models.fastpitch.fastpitch.data_function import (BetaBinomialInterpolator,
                                                      TTSCollate, batch_to_gpu)
from models.fastpitch.fastpitch.loss_function import FastPitchLoss
from models.fastpitch.fastpitch.model import FastPitch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import get_config, progbar, read_lines_from_file
from utils.audio import MelSpectrogram

# %%

device = 'cuda'

def remove_silence(energy_per_frame: torch.Tensor, thresh: float = -10.0):
    keep = energy_per_frame > thresh
    # keep silence at the end
    i = keep.size(0)-1
    while not keep[i] and i > 0:
        keep[i] = True
        i -= 1
    return keep

def normalize_pitch(pitch, mean=130.05478, std=22.86267):
    zeros = (pitch == 0.0)
    pitch -= mean
    pitch /= std
    pitch[zeros] = 0.0
    return pitch

class ArabDataset(Dataset):
    def __init__(self, txtpath='./data/train_phon.txt',
                 wavpath='G:/data/arabic-speech-corpus/wav_new',
                 f0_dict_path='./data/pitch_dict2.pt',
                 cache=False):
        super().__init__()

        self.mel_fn = MelSpectrogram()
        self.wav_path = wavpath
        self.cache = cache

        lines = read_lines_from_file(txtpath)
        
        self.f0_dict = torch.load(f0_dict_path)
        self.betabinomial_interpolator = BetaBinomialInterpolator()

        phoneme_mel_pitch_list = []

        for line in progbar(lines):
            fname, phonemes = line.split('" "')
            fname, phonemes = fname[1:], phonemes[:-1]

            tokens = text.phonemes_to_tokens(phonemes, append_space=False)
            token_ids = text.tokens_to_ids(tokens)
            fpath = os.path.join(self.wav_path, fname)

            if not os.path.exists(fpath):
                print(f"{fpath} does not exist")
                continue

            pitch_mel = self.f0_dict[fname]

            if self.cache:
                mel_log = self._get_mel_from_fpath(fpath)
                phoneme_mel_pitch_list.append(
                    (torch.LongTensor(token_ids), mel_log, pitch_mel))
            else:
                phoneme_mel_pitch_list.append((torch.LongTensor(token_ids), fpath, pitch_mel))

        self.data = phoneme_mel_pitch_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.cache:
            return self.data[idx]

        phonemes, fpath, pitch_mel = self.data[idx]
        
        wave, _ = torchaudio.load(fpath)

        mel_raw = self.mel_fn(wave)
        mel_log = mel_raw.clamp_min(1e-5).log().squeeze()
        keep = remove_silence(mel_log.mean(0))

        mel_log = mel_log[:,keep]
        pitch_mel = normalize_pitch(pitch_mel[:,keep])
        energy = torch.norm(mel_log.float(), dim=0, p=2)
        attn_prior = torch.from_numpy(
            self.betabinomial_interpolator(mel_log.size(1), len(phonemes)))

        speaker = None
        return (phonemes, mel_log, len(phonemes), pitch_mel, energy, speaker, attn_prior,
                fpath)

class DynBatchDataset(ArabDataset):
    def __init__(self, txtpath='./data/train_phon.txt',
                 wavpath='G:/data/arabic-speech-corpus/wav_new',
                 f0_dict_path='./data/pitch_dict2.pt',
                 cache=False):
        super().__init__(txtpath=txtpath, wavpath=wavpath, 
                         f0_dict_path=f0_dict_path,
                         cache=cache)

        self.max_lens = [0, 1000, 1300, 1850, 30000]
        self.b_sizes = [10, 8, 6, 4]

        self.id_batches = []
        self.shuffle()

    def shuffle(self):        
        lens = [x[2].size(1) for x in self.data]


        ids_per_bs = {b: [] for b in self.b_sizes}

        for i, mel_len in enumerate(lens):
            b_idx = next(i for i in range(len(self.max_lens)-1) if self.max_lens[i] <= mel_len < self.max_lens[i+1])
            ids_per_bs[self.b_sizes[b_idx]].append(i)

        id_batches = []

        for bs, ids in ids_per_bs.items():
            np.random.shuffle(ids)
            ids_chnk = [ids[i:i+bs] for i in range(0,len(ids),bs)]
            id_batches += ids_chnk

        self.id_batches = id_batches

    def __len__(self):
        return len(self.id_batches)

    def __getitem__(self, idx):
        batch = [super(DynBatchDataset, self).__getitem__(idx) for idx in self.id_batches[idx]]
        return batch


config = get_config('./configs/nawar.yaml')

# train_dataset = ArabDataset(config.train_labels, config.train_wavs_path,
#                             cache=config.cache_dataset)
train_dataset = DynBatchDataset(config.train_labels, config.train_wavs_path,
                            cache=config.cache_dataset)

# %%


collate_fn = TTSCollate()

config.batch_size = 1
sampler, shuffle, drop_last = None, True, True
train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          collate_fn=lambda x: collate_fn(x[0]),
                          shuffle=shuffle, drop_last=drop_last,
                          sampler=sampler)
# %%

config.restore_model = ''
#config.restore_model = './checkpoints/fp0/states.pth'

if config.restore_model != '':
    state_dicts = torch.load(config.restore_model)
    net_config = state_dicts['config']
else:
    state_dicts = {}
    # from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/fastpitch__pyt_ckpt
    model_sd = torch.load('G:/models/fastpitch/nvidia_fastpitch_210824+cfg.pt')
    state_dicts['model'] = {k.removeprefix('module.'): v for k,v in model_sd['state_dict'].items()}


model = FastPitch(**net_config)
model = model.cuda()
model.train();

criterion = FastPitchLoss()
attention_kl_loss = AttentionBinarizationLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, 
                              weight_decay=1e-6)

writer = SummaryWriter('./logs/logs_fp/exp_gan_10')

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

# %%
# resume from existing checkpoint
n_epoch, n_iter = 0, 0

model.load_state_dict(state_dicts['model'])   
if 'optim' in state_dicts:
    optimizer.load_state_dict(state_dicts['optim'])
if 'epoch' in state_dicts:
    n_epoch = state_dicts['epoch']
if 'iter' in state_dicts:
    n_iter = state_dicts['iter']

# %%

model.train()

for epoch in range(100):
    train_dataset.shuffle()
    for batch in train_loader:

        x, y, num_frames = batch_to_gpu(batch)

        y_pred = model(x)  

        mel_out, _, _, _, _, _, _, _, attn_soft, attn_hard, _, _ = y_pred
        _, _, mel_padded, output_lengths, *_ = x
      
        loss, meta = criterion(y_pred, y)    
        
        binarization_loss = attention_kl_loss(attn_hard, attn_soft)
        meta['kl_loss'] = binarization_loss.clone().detach()
        loss += 1.0 * binarization_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1000.)
        optimizer.step()

        print(f"loss: {meta['loss'].item()} gnorm: {grad_norm}")

        n_iter += 1

        for k, v in meta.items():
            writer.add_scalar(f'train/{k}', v.item(), n_iter)

# %%
def save_states(fname, model, optimizer,
                n_iter, epoch, net_config):
    torch.save({'model': model.state_dict(),                              
                'optim': optimizer.state_dict(),            
                'epoch': epoch,
                'iter': n_iter,
                'config': net_config,
                },
               f'./checkpoints/fp0/{fname}_{n_iter}')


save_states(f'states.pth', model,
            optimizer, n_iter, epoch, net_config, config)

# %%





# %%

idx = 0
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.imshow(y_pred[0][idx,:y[2][idx],:].detach().cpu().t(), aspect='auto', origin='lower')
ax2.imshow(y[0][idx,:,:y[2][idx]].detach().cpu(), aspect='auto', origin='lower')


# %%

import sounddevice as sd
from vocoder import load_hifigan

model.eval()
vocoder = load_hifigan(config.vocoder_state_path, config.vocoder_config_path)
vocoder = vocoder.cuda()

# %%

with torch.inference_mode():
    (mel_out, dec_lens, dur_pred, 
    pitch_pred, energy_pred) = model.infer(x[0][2:3])

    wave = vocoder(mel_out[0])

mel_out.shape
plt.imshow(mel_out[0].cpu(), aspect='auto', origin='lower')

plt.plot(wave[0].cpu())

sd.play(0.3*wave[0].cpu(), 22050)

# %%

test_dataset = ArabDataset(config.test_labels, config.test_wavs_path,
                            cache=config.cache_dataset, f0_dict_path='./data/wav_f0_dict_test.pt')

# %%

model.eval()

(phonemes, mel_log, len_phonemes, 
 pitch_mel, energy, speaker, attn_prior,
 fpath) = test_dataset[2]


with torch.inference_mode():
    (mel_out, dec_lens, dur_pred, 
    pitch_pred, energy_pred) = model.infer(phonemes[None,:].cuda(), pace=1)

    wave = vocoder(mel_out[0])

sd.play(0.3*wave[0].cpu(), 22050)

plt.plot(pitch_pred.cpu()[0,0])

