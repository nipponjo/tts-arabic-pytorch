import torch
import torch.nn as nn
import torch.nn.functional as F


# described in: "One TTS Alignment To Rule Them All" (https://arxiv.org/abs/2108.10447)
# adapted from: https://github.com/NVIDIA/radtts/blob/07759cd474458f46db45cab975a85ba21b7fee0a/loss.py#L111
class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, in_lens, out_lens):
        attn_logprob = (attn_logprob+1e-12)[:,None].log()
        
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(
            input=attn_logprob, pad=(1, 0, 0, 0, 0, 0, 0, 0),
            value=self.blank_logprob)
        cost_total = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid]+1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                :query_lens[bid], :, :key_lens[bid]+1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            ctc_cost = self.CTCLoss(curr_logprob, target_seq,
                                    input_lengths=query_lens[bid:bid+1],
                                    target_lengths=key_lens[bid:bid+1])
            cost_total += ctc_cost
        cost = cost_total/attn_logprob.shape[0]
        return cost


def get_mask_from_lengths(lengths, x):
    max_len = x.shape[-1]
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = ids < lengths.unsqueeze(1)
    return mask


# adapted from: https://github.com/NVIDIA/NeMo/blob/557c4b7ae766faf050374e6b9a862e2e67385b10/nemo/collections/tts/losses/tacotron2loss.py#L23
class Tacotron2Loss(nn.Module):
    def __init__(self, 
                 mel_loss_scale=1.0, 
                 attn_loss_scale=0.1):
        super().__init__()
        self.mel_loss_scale = mel_loss_scale
        self.attn_loss_scale = attn_loss_scale
        self.attn_ctc_loss = AttentionCTCLoss()

    def forward(self,
                spec_pred_dec,
                spec_pred_postnet,
                gate_pred,
                spec_target,
                alignments,
                input_lengths,
                spec_target_len,                
                pad_value=0):

        # Make the gate target
        max_len = spec_target.shape[2]
        gate_target = torch.zeros(spec_target_len.shape[0], max_len)
        gate_target = gate_target.type_as(gate_pred)
        for i, length in enumerate(spec_target_len):
            gate_target[i, length.data - 1:] = 1

        spec_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        max_len = spec_target.shape[2]

        if max_len < spec_pred_dec.shape[2]:
            # Predicted len is larger than reference
            # Need to slice
            spec_pred_dec = spec_pred_dec.narrow(2, 0, max_len)
            spec_pred_postnet = spec_pred_postnet.narrow(2, 0, max_len)
            gate_pred = gate_pred.narrow(1, 0, max_len).contiguous()
        elif max_len > spec_pred_dec.shape[2]:
            # Need to do padding
            pad_amount = max_len - spec_pred_dec.shape[2]
            spec_pred_dec = torch.nn.functional.pad(
                spec_pred_dec, (0, pad_amount), value=pad_value)
            spec_pred_postnet = torch.nn.functional.pad(
                spec_pred_postnet, (0, pad_amount), value=pad_value)
            gate_pred = torch.nn.functional.pad(
                gate_pred, (0, pad_amount), value=1e3)

        mask = ~get_mask_from_lengths(spec_target_len, spec_pred_dec)
        mask = mask.expand(spec_target.shape[1], mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        spec_pred_dec.data.masked_fill_(mask, pad_value)
        spec_pred_postnet.data.masked_fill_(mask, pad_value)
        gate_pred.data.masked_fill_(mask[:, 0, :], 1e3)

        gate_pred = gate_pred.view(-1, 1)
        rnn_mel_loss = torch.nn.functional.mse_loss(spec_pred_dec, spec_target)
        postnet_mel_loss = torch.nn.functional.mse_loss(
            spec_pred_postnet, spec_target)
        gate_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            gate_pred, gate_target)
        
        loss_attn = self.attn_ctc_loss(alignments, input_lengths, spec_target_len)
  
        meta = {
            'mel_loss_rnn': rnn_mel_loss.clone().detach(),
            'mel_loss_postnet': postnet_mel_loss.clone().detach(),
            'gate_loss': gate_loss.clone().detach(),
            'attn_loss': loss_attn.clone().detach(),
        }    

        loss = self.mel_loss_scale * rnn_mel_loss \
                + self.mel_loss_scale * postnet_mel_loss \
                + gate_loss \
                + self.attn_loss_scale * loss_attn

        return loss, meta
    


# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.



# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.