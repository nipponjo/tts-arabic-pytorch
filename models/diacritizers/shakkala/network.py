import torch
import torch.nn as nn
from .lstm_hsm import LSTMHardSigmoid
from . import encode, decode
from typing import Union, List


class Shakkala(nn.Module):
    def __init__(self, 
                 dim_input: int=149, 
                 dim_output: int=28,
                 sd_path: str=None):
        super().__init__()
        self.emb_input = nn.Embedding(dim_input, 288)

        self.lstm0 = LSTMHardSigmoid(288, hidden_size=288, bidirectional=True, batch_first=True)
        self.bn0 = nn.BatchNorm1d(576, momentum=0.01, eps=0.001)
        self.lstm1 = LSTMHardSigmoid(576, hidden_size=144, bidirectional=True, batch_first=True)
        self.lstm2 = LSTMHardSigmoid(288, hidden_size=96, bidirectional=True, batch_first=True)

        self.dense0 = nn.Linear(192, dim_output)

        self.eval()
        self.max_sentence = None

        if sd_path is not None:
            self.load_state_dict(torch.load(sd_path))
    
    def forward(self, x: torch.Tensor):
        x = self.emb_input(x)

        x, _ = self.lstm0(x)
        x = self.bn0(x.transpose(1,2)).transpose(1,2)   
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.dense0(x)
        x = nn.Softmax(dim=-1)(x)

        return x
    
    @torch.inference_mode()
    def infer(self, x: torch.Tensor):
        return self.forward(x)    

    def _predict_list(self, input_list: List[str], return_probs: bool=False):
        output_list = []
        probs_list = []
        for input_text in input_list:
            if return_probs:
                output_text, probs = self._predict_single(input_text, return_probs=True)
                output_list.append(output_text)
                probs_list.append(probs)
            else:
                output_list.append(self._predict_single(input_text))

        if return_probs:
            return output_list, return_probs
        
        return output_list
    
    def _predict_single(self, input_text: str, return_probs: bool=False):
        input_ids_pad, input_letters_ids = encode(input_text, self.max_sentence)
        input = torch.LongTensor(input_ids_pad)[None].to(self.emb_input.weight.device)
        probs = self.infer(input).cpu()
        output = decode(probs, input_text, input_letters_ids)

        if return_probs:
            return output, probs
        
        return output

    def predict(self, input: Union[str, List[str]], return_probs: bool=False):
        if isinstance(input, str):
            return self._predict_single(input, return_probs=return_probs)        
        
        return self._predict_list(input, return_probs=return_probs)
