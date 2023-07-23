import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encode, decode
from typing import Union, List


class Shakkelha(nn.Module):
    def __init__(self, 
                 dim_input: int=91, 
                 dim_output: int=19,
                 sd_path: str=None):        
        super().__init__()
        self.emb0 = nn.Embedding(dim_input, 25)

        self.lstm0 = nn.LSTM(25, 256, batch_first=True, bidirectional=True)
        self.lstm1 = nn.LSTM(512, 256, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(p=0.5)

        self.dense0 = nn.Linear(512, 512)
        self.dense1 = nn.Linear(512, 512)
        self.dense2 = nn.Linear(512, dim_output)

        self.eval()

        if sd_path is not None:
            self.load_state_dict(torch.load(sd_path))
    
    def forward(self, x: torch.Tensor):
        x = self.emb0(x)
  
        x, _ = self.lstm0(x) 
        x = self.dropout(x)
        x, _ = self.lstm1(x)
        x = self.dropout(x)   

        x = F.relu(self.dense0(x)) 
        x = F.relu(self.dense1(x))  
        x = F.softmax(self.dense2(x), dim=-1)
   
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
            return output_list, probs_list
        
        return output_list
    
    def _predict_single(self, input_text: str, return_probs: bool=False):
        ids = encode(input_text)
        input = torch.LongTensor(ids)[None].to(self.emb0.weight.device)
        probs = self.infer(input).cpu()
        output = decode(probs, input_text)

        if return_probs:
            return output, probs
        
        return output

    def predict(self, input: Union[str, List[str]], return_probs: bool=False):
        if isinstance(input, str):
            return self._predict_single(input, return_probs=return_probs)        
        
        return self._predict_list(input, return_probs=return_probs)