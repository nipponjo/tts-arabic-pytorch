import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn


class MelSpectrogram(nn.Module):
    def __init__(self, 
                 sample_rate: int = 22050,
                 n_fft: int = 1024,
                 win_length: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 80,
                 f_min: float = 0,
                 f_max: float = 8000.0,
                 norm: str = 'slaney',
                 center: bool = False
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
        spec_lin = torch.stft(x_pad, self.n_fft,
                              self.hop_length,
                              self.win_length,
                              self.window_fn,
                              center=self.center,
                              return_complex=True) # [B, F, T]
        spec_mag = spec_lin.abs().pow_(2).add_(1e-9).sqrt_()
        spec_mel = torch.matmul(self.mel_basis, spec_mag) # [B, mels, T]
        return spec_mel
