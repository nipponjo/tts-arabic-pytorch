import torch.nn as nn
import torch.nn.functional as F


class Tacotron2Loss(nn.Module):
    def __init__(self, 
                 mel_loss_scale=1.0):
        super().__init__()
        self.mel_loss_scale = mel_loss_scale

    def forward(self, 
                mel_out, 
                mel_out_postnet, 
                mel_padded,
                gate_out,
                gate_padded):
        
        rnn_mel_loss = F.mse_loss(mel_out, mel_padded)
        postnet_mel_loss = F.mse_loss(mel_out_postnet, mel_padded)            
        gate_loss = F.binary_cross_entropy_with_logits(
            gate_out, gate_padded)
        
        meta = {
            'mel_loss_rnn': rnn_mel_loss.clone().detach(),
            'mel_loss_postnet': postnet_mel_loss.clone().detach(),
            'gate_loss': gate_loss.clone().detach(),           
        }    

        loss = self.mel_loss_scale * rnn_mel_loss \
        + self.mel_loss_scale * postnet_mel_loss \
        + gate_loss 

        return loss, meta
   
