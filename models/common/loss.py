import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List


def extract_chunks(A: Tensor, 
                   ofx: Tensor, 
                   mel_ids: Optional[Tensor] = None, 
                   chunk_len: int = 128):
    """
    Args:
        A (Tensor): spectrograms [B, F, T]
        ofx (Tensor): offsets [num_chunks,]
        mel_ids (Tensor): [num_chunks,]
    Returns:
        chunks (Tensor): [num_chunks, F, chunk_len]
    """
    ids = torch.arange(0, chunk_len, device=A.device)[None,:].repeat(len(mel_ids), 1) + ofx[:,None]

    if mel_ids is None:
        mel_ids = torch.arange(0, A.size(0), device=A.device)[:,None] * A.size(2)
    ids = ids + mel_ids[:,None] * A.size(2)

    chunks = A.transpose(0, 1).flatten(1)[:, ids.long()].transpose(0, 1)
    return chunks


def calc_feature_match_loss(fmaps_gen: List[Tensor],
                            fmaps_org: List[Tensor]
                            ):
    
    loss_fmatch = 0.
    for (fmap_gen, fmap_org) in zip(fmaps_gen, fmaps_org):
        fmap_org.detach_()
        loss_fmatch += (fmap_gen - fmap_org).abs().mean()

    loss_fmatch = loss_fmatch / len(fmaps_gen)
    return loss_fmatch


class Conv2DSpectralNorm(nn.Conv2d):
    """Convolution layer that applies Spectral Normalization before every call."""

    def __init__(self, cnum_in: int, cnum_out: int, 
                 kernel_size: int, stride: int, padding: int = 0, 
                 n_iter: int = 1, eps: float = 1e-12, 
                 bias: bool = True):
        super().__init__(cnum_in,
                         cnum_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
        self.register_buffer("weight_u", torch.empty(self.weight.size(0), 1))
        nn.init.trunc_normal_(self.weight_u)
        self.n_iter = n_iter
        self.eps = eps

    def l2_norm(self, x):
        return F.normalize(x, p=2, dim=0, eps=self.eps)

    def forward(self, x):

        weight_orig = self.weight.flatten(1).detach()

        for _ in range(self.n_iter):
            v = self.l2_norm(weight_orig.t() @ self.weight_u)
            self.weight_u = self.l2_norm(weight_orig @ v)

        sigma = self.weight_u.t() @ weight_orig @ v
        self.weight.data.div_(sigma)

        x = super().forward(x)

        return x


class DConv(nn.Module):
    def __init__(self, cnum_in,
                 cnum_out, ksize=5, stride=2, padding='auto'):
        super().__init__()
        padding = (ksize-1)//2 if padding == 'auto' else padding
        self.conv_sn = Conv2DSpectralNorm(
            cnum_in, cnum_out, ksize, stride, padding)
        #self.conv_sn = spectral_norm(nn.Conv2d(cnum_in, cnum_out, ksize, stride, padding))
        self.leaky = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv_sn(x)
        x = self.leaky(x)
        return x


class PatchDiscriminator(nn.Module):
    def __init__(self, cnum_in, cnum):
        super().__init__()
        self.conv1 = DConv(cnum_in, cnum)
        self.conv2 = DConv(cnum, 2*cnum)
        self.conv3 = DConv(2*cnum, 4*cnum)
        self.conv4 = DConv(4*cnum, 4*cnum)
        self.conv5 = DConv(4*cnum, 4*cnum)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x = nn.Flatten()(x5)

        return x, [x1, x2, x3, x4]