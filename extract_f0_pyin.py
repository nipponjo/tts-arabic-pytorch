# %%
import os
import torch
import librosa
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from utils.audio import MelSpectrogram
from utils import write_lines_to_file, scandir

# %% CONFIG

waves_dir = 'I:/tts/3arabiyya/arabic-speech-corpus/wav_red'
pitches_dir = 'I:/tts/3arabiyya/arabic-speech-corpus/wav_red/pitches_pyin'


fmin = librosa.note_to_hz('C2') # C1: 32.70 C2: 65.41 C3: 130.81 C4: 261.63
fmax = librosa.note_to_hz('C5') # C5: 523.25 C6: 1046.50 C7: 2093.00

mel_trf = MelSpectrogram()

# %% SCAN WAVE FOLDER

wave_filepaths = scandir(waves_dir)
print(f"{len(wave_filepaths)} wave files found @ {waves_dir}")

# %% extract pitch (f0) values


for i, wave_filepath in tqdm(enumerate(wave_filepaths), 
                             total=len(wave_filepaths)):    

    wave_relpath = os.path.relpath(wave_filepath, waves_dir)
    pitch_filepath = f"{os.path.join(pitches_dir, wave_relpath)}.pth"
    
    if os.path.exists(pitch_filepath): continue
    pitch_dir = os.path.dirname(pitch_filepath)
    os.makedirs(pitch_dir, exist_ok=True)
    
    wav, sr = librosa.load(wave_filepath, sr=mel_trf.sample_rate)

    mel_spec = mel_trf(torch.tensor(wav)[None])[0] # [mel_bands, T]

    # estimate pitch
    pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
        wav, sr=mel_trf.sample_rate,
        fmin=fmin, fmax=fmax,
        frame_length=mel_trf.win_length,
        hop_length=mel_trf.hop_length)

    pitch_mel = np.where(np.isnan(pitch_mel), 0., pitch_mel) # set nan to zero
    pitch_mel = torch.from_numpy(pitch_mel)
    pitch_mel = F.pad(pitch_mel, (0, mel_spec.size(1) - pitch_mel.size(0))) # pad to mel length

    # save pitch
    torch.save(pitch_mel, pitch_filepath)


# %% calculate pitch mean and std

from datetime import datetime

pitch_filepaths = scandir(pitches_dir, extensions=('.pth'))

rmean = 0
rvar = 0
ndata = 0

for pitch_filepath in pitch_filepaths: 
    
    pitch_mel = torch.load(pitch_filepath)    
      
    pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
    
    pitch_mel_ = pitch_mel[pitch_mel > 1]
    if len(pitch_mel_) == 0: continue
    
    p_mean = np.mean(pitch_mel_)
    p_var = np.var(pitch_mel_)
    p_len = len(pitch_mel_)

    rvar = ((ndata-1)*rvar + (p_len-1)*p_var) / (ndata + p_len - 1) + \
            ndata*p_len*(p_mean - rmean)**2 / ((ndata + p_len)*(ndata + p_len - 1))
    
    rmean = (p_len*p_mean + ndata*rmean) / (p_len + ndata)

    ndata += p_len

mean, std = rmean, np.sqrt(rvar)
print('mean', mean)
print('std', std)

timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

write_lines_to_file(path=f'./data/mean_std_{timestamp}.txt', 
                    lines=[
                        f"dir: {pitches_dir}",
                        f"nfiles: {len(pitch_filepaths)}",
                        f"method: pyin",
                        f"mean: {mean}", 
                        f"std: {std}"])

# %%