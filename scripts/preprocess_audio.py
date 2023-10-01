# %%
import os
import torch
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm

# %%

wav_path = 'G:/data/arabic-speech-corpus/wav'
wav_new_path = 'G:/data/arabic-speech-corpus/wav_new'

sr_target = 22050
silence_audio_size = 256 * 3

device = 'cuda'

wav_fpaths = [f.path for f in os.scandir(wav_path) if f.path.endswith('.wav')]
# waves = make_dataset_from_subdirs(wavs_path)

if not os.path.exists(wav_new_path):
    os.makedirs(wav_new_path)
    print(f"Created folder @ {wav_new_path}")

# %%

for wav_fpath in tqdm(wav_fpaths):

    fname = os.path.basename(wav_fpath)

    fpath = os.path.join(wav_path, fname)
    wave, sr = torchaudio.load(fpath)
    
    if sr != sr_target:
        wave = wave.to(device)
        wave = torchaudio.functional.resample(wave, sr, sr_target, 
                                              lowpass_filter_width=1024)

    wave_ = wave[0].cpu().numpy()
    wave_ = wave_ / np.abs(wave_).max() * 0.999
    wave_ = librosa.effects.trim(
        wave_, top_db=23, frame_length=1024, hop_length=256)[0]
    wave_ = np.append(wave_, [0.]*silence_audio_size)

    torchaudio.save(f'{wav_new_path}/{fname}',
                    torch.Tensor(wave_).unsqueeze(0), sr_target)


# %%
