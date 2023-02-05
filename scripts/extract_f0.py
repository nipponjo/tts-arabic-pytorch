# %%
import librosa
import torch
import glob
import os
from tqdm import tqdm
import numpy as np
from utils.audio import MelSpectrogram
import torch.nn.functional as F

# %%

waves = glob.glob('G:/data/arabic-speech-corpus/wav_new/*.wav')

mel_trf = MelSpectrogram()

wav_f0_dict = {}

# %%

for i, wav_path in tqdm(enumerate(waves)):
    wav, sr = librosa.load(wav_path, sr=22050)

    wav_name = os.path.basename(wav_path)
    if wav_name in wav_f0_dict:
        continue

    pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
        wav, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'), frame_length=1024)

    mel_log = mel_trf(torch.tensor(wav)[None,:])

    pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
    pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
    pitch_mel = F.pad(pitch_mel, (0, mel_log.size(-1) - pitch_mel.size(1)))

    wav_f0_dict[wav_name] = pitch_mel

    if i % 10 == 0:
        torch.save(wav_f0_dict, './data/wav_f0_dict.pt')

torch.save(wav_f0_dict, './data/wav_f0_dict.pt')



# %%
import torch
import numpy as np

wav_f0_dict = torch.load('./data/pitch_dict2.pt')

rmean = 0
rvar = 0
ndata = 0

for pitch_mel in wav_f0_dict.values():   
    pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
    
    pitch_mel_ = pitch_mel[pitch_mel > 1]
    p_mean = np.mean(pitch_mel_)
    p_var = np.var(pitch_mel_)
    p_len = len(pitch_mel_)

    rvar = ((ndata-1)*rvar + (p_len-1)*p_var) / (ndata + p_len - 1) + \
            ndata*p_len*(p_mean - rmean)**2 / ((ndata + p_len)*(ndata + p_len - 1))
    
    rmean = (p_len*p_mean + ndata*rmean) / (p_len + ndata)

    ndata += p_len

print('mean ', rmean)
print('std ', np.sqrt(rvar))

# %%
