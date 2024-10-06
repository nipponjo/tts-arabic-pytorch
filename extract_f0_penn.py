# %%
import os
import torch
import torchaudio
import numpy as np

import librosa
import penn

from tqdm import tqdm
from utils import scandir, write_lines_to_file
from utils.audio import MelSpectrogram

# %% CONFIG

waves_dir = 'I:/tts/3arabiyya/arabic-speech-corpus/wav_red'
pitches_dir = 'I:/tts/3arabiyya/arabic-speech-corpus/wav_red/pitches_penn'


fmin = librosa.note_to_hz('C2') # C1: 32.70 C2: 65.41 C3: 130.81 C4: 261.63
fmax = librosa.note_to_hz('C5') # C5: 523.25 C6: 1046.50 C7: 2093.00

mel_trf = MelSpectrogram()

# %% SCAN WAVE FOLDER

wave_filepaths = scandir(waves_dir)
print(f"{len(wave_filepaths)} wave files found @ {waves_dir}")


# PENN PARAMS
hopsize = .01
gpu = 0 
batch_size = 1024
checkpoint = None
center = 'half-hop'
interp_unvoiced_at = None


def infer_pitch(wav, sr, thr=0.5, sr8k=True, batch_size=1024):
    
    mel_spec = mel_trf(wav)
    audio = torchaudio.functional.resample(wav, sr, 16000)

    pitch, periodicity = penn.from_audio(
        audio,
        8000 if sr8k else 16000, 
        hopsize=hopsize,
        fmin=fmin, fmax=fmax, 
        checkpoint=checkpoint,
        batch_size=batch_size, 
        center = center,
        interp_unvoiced_at=interp_unvoiced_at, 
        gpu=gpu)


    pitch_nn = pitch[0]
    probs_nn = periodicity[0]

    pitch_nn[probs_nn <= thr] = torch.nan

    tar_size = mel_spec.size(2)
    pitch_res = torch.nn.functional.interpolate(pitch_nn[None,None], size=tar_size)
    if sr8k: pitch_res *= 2

    return pitch_res[0]

if gpu is not None:
    mel_trf.to(f'cuda:{gpu}', non_blocking=True)

# %%

for i, wave_filepath in tqdm(enumerate(wave_filepaths), 
                             total=len(wave_filepaths)):    

    wave_relpath = os.path.relpath(wave_filepath, waves_dir)
    pitch_filepath = f"{os.path.join(pitches_dir, wave_relpath)}.pth"
    
    if os.path.exists(pitch_filepath): continue
    pitch_dir = os.path.dirname(pitch_filepath)
    os.makedirs(pitch_dir, exist_ok=True)
    
    wav, sr = librosa.load(wave_filepath, sr=mel_trf.sample_rate)
    wav = torch.from_numpy(wav)
    if gpu is not None:
        wav = wav.to(f'cuda:{gpu}', non_blocking=True)
    
    mel_spec = mel_trf(torch.tensor(wav)[None])[0] # [mel_bands, T]

    # estimate pitch
    pitch_penn = infer_pitch(wav[None], sr, thr=0.5, sr8k=False)
    pitch_penn = torch.where(torch.isnan(pitch_penn), 0.0, pitch_penn)
    
    pitch_penn = pitch_penn.to(device='cpu', non_blocking=True)
    
    # save pitch
    torch.save(pitch_penn, pitch_filepath)



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
                        f"method: penn",
                        f"mean: {mean}", 
                        f"std: {std}"])

# %%