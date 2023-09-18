# %%
import os
import torch
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm

# %%

txtpath = './data/train_phon.txt'
wav_path = 'G:/data/arabic-speech-corpus/wav'
wav_new_path = 'G:/data/arabic-speech-corpus/wav_new'

# txtpath='./data/test_phon.txt'
# wav_path='G:/data/arabic-speech-corpus/test set/wav'
# wav_new_path='G:/data/arabic-speech-corpus/test set/wav_new/'

# %%

sr_target = 22050
silence_audio_size = 256 * 3


lines = open(txtpath).readlines()

for line in tqdm(lines):

    fname = line.split('" "')[0][:-1]

    fpath = os.path.join(wav_path, fname)
    wave, sr = torchaudio.load(fpath)
    
    if sr != sr_target:
        wave = torchaudio.functional.resample(wave, sr, sr_target, 
                                              lowpass_filter_width=1024)

    wave_ = wave[0].numpy()
    wave_ = wave_ / np.abs(wave_).max() * 0.999
    wave_ = librosa.effects.trim(
        wave_, top_db=23, frame_length=1024, hop_length=256)[0]
    wave_ = np.append(wave_, [0.]*silence_audio_size)

    torchaudio.save(f'{wav_new_path}/{fname}',
                    torch.Tensor(wave_).unsqueeze(0), sr_target)


# %%
