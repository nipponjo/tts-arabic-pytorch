import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn


class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate=22050,
                 n_fft=1024,
                 win_length=1024,
                 hop_length=256,
                 n_mels=80,
                 f_min=0,
                 f_max=8000.0,
                 norm='slaney',
                 center=False
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center

        self.pad_length = int((n_fft - hop_length)/2)

        mel_basis = torch.Tensor(librosa_mel_fn(sr=sample_rate,
                                                n_fft=n_fft, n_mels=n_mels,
                                                fmin=f_min, fmax=f_max, 
                                                norm=norm))
        window_fn = torch.hann_window(win_length)
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('window_fn', window_fn)

    def forward(self, x):
        x_pad = torch.nn.functional.pad(
            x, (self.pad_length, self.pad_length), mode='reflect')
        spec_ta = torch.stft(x_pad, self.n_fft,
                             self.hop_length,
                             self.win_length,
                             self.window_fn,
                             center=self.center,
                             return_complex=False)
        spec_ta = torch.sqrt(spec_ta.pow(2).sum(-1) + 1e-9)
        mel_ta2 = torch.matmul(self.mel_basis, spec_ta)
        return mel_ta2
