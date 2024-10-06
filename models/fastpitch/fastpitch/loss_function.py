# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# from common.utils import mask_from_lens
from .attn_loss_function import AttentionCTCLoss

def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


class FastPitchLoss(nn.Module):
    def __init__(self, 
                 mel_loss_scale=1.0,
                #  mel_lin_loss_scale=1.0,
                 mel_loss_over_scale=1.0,
                 mel_loss_under_scale=1.0,
                 dur_predictor_loss_scale=1.0,
                 dur_loss_toofast_scale=1.0,
                 dur_loss_tooslow_scale=1.0,
                 pitch_predictor_loss_scale=1.0, attn_loss_scale=1.0,
                 energy_predictor_loss_scale=0.1):
        super(FastPitchLoss, self).__init__()

        self.mel_loss_scale = mel_loss_scale

        self.mel_loss_over_scale = mel_loss_over_scale
        self.mel_loss_under_scale = mel_loss_under_scale
        # self.mel_lin_loss_scale = mel_lin_loss_scale

        self.dur_predictor_loss_scale = dur_predictor_loss_scale
        self.dur_loss_toofast_scale = dur_loss_toofast_scale
        self.dur_loss_tooslow_scale = dur_loss_tooslow_scale

        self.pitch_predictor_loss_scale = pitch_predictor_loss_scale
        self.energy_predictor_loss_scale = energy_predictor_loss_scale
        self.attn_loss_scale = attn_loss_scale
        self.attn_ctc_loss = AttentionCTCLoss()

    def forward(self, model_out, targets, is_training=True, meta_agg='mean'):
        (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred, pitch_tgt,
         energy_pred, energy_tgt, attn_soft, attn_hard, attn_dur,
         attn_logprob) = model_out

        (mel_tgt, in_lens, out_lens) = targets

        dur_tgt = attn_dur
        dur_lens = in_lens

        mel_tgt.requires_grad = False
        # (B,H,T) => (B,T,H)
        mel_tgt = mel_tgt.transpose(1, 2)

        dur_mask = mask_from_lens(dur_lens, max_len=dur_tgt.size(1))
        log_dur_tgt = torch.log(dur_tgt.float() + 1)
        loss_fn = F.mse_loss
        dur_pred_loss = loss_fn(log_dur_pred, log_dur_tgt, reduction='none')
        w_dur_loss = (dur_pred > dur_tgt)*self.dur_loss_tooslow_scale \
                   + (dur_pred < dur_tgt)*self.dur_loss_toofast_scale
        dur_pred_loss = (0.5*w_dur_loss * dur_pred_loss * dur_mask).sum() / dur_mask.sum()

        ldiff = mel_tgt.size(1) - mel_out.size(1)
        mel_out = F.pad(mel_out, (0, 0, 0, ldiff, 0, 0), value=0.0)
        mel_mask = mel_tgt.ne(0).float()

        # loss_fn = F.mse_loss
        # mel_loss = loss_fn(mel_out, mel_tgt, reduction='none')
        # mel_loss = (mel_loss * mel_mask).sum() / mel_mask.sum()

        error = mel_out - mel_tgt
        error_out_over = error*(error>0) # pred over target
        error_out_under = error*(error<0)

        # mel_loss = (self.mel_loss_over_scale*error_out_over.square() 
        #                 + self.mel_loss_under_scale*error_out_under.square()) 
        mel_loss_over = self.mel_loss_over_scale*(error_out_over.square())
        mel_loss_under = self.mel_loss_under_scale*(error_out_under.square())
               
        # mel_lin_loss = loss_fn(mel_out.exp(), mel_tgt.exp(), reduction='none')
        mask_sum = mel_mask.sum()
        mel_loss_over = (mel_loss_over * mel_mask).sum() / mask_sum
        mel_loss_under  = (mel_loss_under * mel_mask).sum() / mask_sum
        # mel_loss = 2*mel_loss / (self.mel_loss_over_scale + self.mel_loss_under_scale)
        mel_loss = 2*(mel_loss_over + mel_loss_under) / (self.mel_loss_over_scale + self.mel_loss_under_scale)


        # loss_fn = F.l1_loss

        # error_lin = mel_out.exp() - mel_tgt.exp()
        # error_out_over = error_lin*(error_lin>0)
        # error_out_under = error_lin*(error_lin<0)

        # mel_lin_loss = (self.mel_loss_over_scale*error_out_over.abs() 
        #                 + self.mel_loss_under_scale*error_out_under.abs())        
        # mel_lin_loss = loss_fn(mel_out.exp(), mel_tgt.exp(), reduction='none')
        # mel_lin_loss = (mel_lin_loss * mel_mask).sum() / mel_mask.sum()
        # mel_lin_loss = 2*mel_lin_loss / (self.mel_loss_over_scale + self.mel_loss_under_scale)

        ldiff = pitch_tgt.size(2) - pitch_pred.size(2)
        pitch_pred = F.pad(pitch_pred, (0, ldiff, 0, 0, 0, 0), value=0.0)
        pitch_loss = F.mse_loss(pitch_tgt, pitch_pred, reduction='none')
        pitch_loss = (pitch_loss * dur_mask.unsqueeze(1)).sum() / dur_mask.sum()

        if energy_pred is not None:
            energy_pred = F.pad(energy_pred, (0, ldiff, 0, 0), value=0.0)
            energy_loss = F.mse_loss(energy_tgt, energy_pred, reduction='none')
            energy_loss = (energy_loss * dur_mask).sum() / dur_mask.sum()
        else:
            energy_loss = 0

        # Attention loss
        attn_loss = self.attn_ctc_loss(attn_logprob, in_lens, out_lens)

        loss = (mel_loss * self.mel_loss_scale
                # + mel_lin_loss * self.mel_lin_loss_scale
                + dur_pred_loss * self.dur_predictor_loss_scale
                + pitch_loss * self.pitch_predictor_loss_scale
                + energy_loss * self.energy_predictor_loss_scale
                + attn_loss * self.attn_loss_scale)

        meta = {
            'loss': loss.clone().detach(),
            'mel_loss': mel_loss.clone().detach(),
            'mel_loss_over': mel_loss_over.clone().detach(),
            'mel_loss_under': mel_loss_under.clone().detach(),
            # 'mel_lin_loss': mel_lin_loss.clone().detach(),
            'duration_predictor_loss': dur_pred_loss.clone().detach(),
            'pitch_loss': pitch_loss.clone().detach(),
            'attn_loss': attn_loss.clone().detach(),
            'dur_error': (torch.abs(dur_pred - dur_tgt).sum()
                          / dur_mask.sum()).detach(),          
        }

        if energy_pred is not None:
            meta['energy_loss'] = energy_loss.clone().detach()

        assert meta_agg in ('sum', 'mean')
        if meta_agg == 'sum':
            bsz = mel_out.size(0)
            meta = {k: v * bsz for k, v in meta.items()}
        return loss, meta
