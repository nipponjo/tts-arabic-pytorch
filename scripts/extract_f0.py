# %%
import os
import torch
import librosa
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from utils.audio import MelSpectrogram
from utils import write_lines_to_file

# %% CONFIG

wavs_path = 'G:/data/arabic-speech-corpus/wav_new'

waves = [f.path for f in os.scandir(wavs_path) if f.path.endswith('.wav')]
print(f"{len(waves)} wave files found at {wavs_path}")

mel_trf = MelSpectrogram()

# %% extract pitch (f0) values

pitch_dict = {}

for i, wav_path in tqdm(enumerate(waves), total=len(waves)):
    wav, sr = librosa.load(wav_path, sr=mel_trf.sample_rate)

    wav_name = os.path.basename(wav_path)
    if wav_name in pitch_dict:
        continue
    mel_spec = mel_trf(torch.tensor(wav)[None])[0] # [mel_bands, T]

    # estimate pitch
    pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
        wav, sr=mel_trf.sample_rate,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=mel_trf.win_length,
        hop_length=mel_trf.hop_length)

    pitch_mel = np.where(np.isnan(pitch_mel), 0., pitch_mel) # set nan to zero
    pitch_mel = torch.from_numpy(pitch_mel)
    pitch_mel = F.pad(pitch_mel, (0, mel_spec.size(1) - pitch_mel.size(0))) # pad to mel length

    pitch_dict[wav_name] = pitch_mel

    if i % 10 == 0: # save intermediate dict
        torch.save(pitch_dict, './data/pitch_dict.pt')

torch.save(pitch_dict, './data/pitch_dict.pt')


# %% calculate pitch mean and std

pitch_dict = torch.load('./data/pitch_dict.pt')

rmean = 0
rvar = 0
ndata = 0

for pitch_mel in pitch_dict.values():   
    pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
    
    pitch_mel_ = pitch_mel[pitch_mel > 1]
    p_mean = np.mean(pitch_mel_)
    p_var = np.var(pitch_mel_)
    p_len = len(pitch_mel_)

    rvar = ((ndata-1)*rvar + (p_len-1)*p_var) / (ndata + p_len - 1) + \
            ndata*p_len*(p_mean - rmean)**2 / ((ndata + p_len)*(ndata + p_len - 1))
    
    rmean = (p_len*p_mean + ndata*rmean) / (p_len + ndata)

    ndata += p_len

mean, std = rmean, np.sqrt(rvar)
print('mean ', mean)
print('std ', std)

write_lines_to_file(path='./data/mean_std.txt', 
                    lines=[f"mean: {mean}", 
                           f"std: {std}"])
