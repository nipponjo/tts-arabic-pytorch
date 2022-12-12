from typing import List, Union

import text
import torch
import torch.nn as nn

from utils import get_basic_config
from vocoder import load_hifigan
from vocoder.hifigan.denoiser import Denoiser

from .fastpitch.model import FastPitch as _FastPitch


def text_collate_fn(batch: List[torch.Tensor]):
    """
    Args:
        batch: List[text_ids]
    Returns:
        text_ids_pad
        input_lens_sorted
        reverse_ids 
    """
    input_lens_sorted, input_sort_ids = torch.sort(
        torch.LongTensor([len(x) for x in batch]), descending=True)
    max_input_len = input_lens_sorted[0]

    text_ids_pad = torch.LongTensor(len(batch), max_input_len)
    text_ids_pad.zero_()
    for i in range(len(input_sort_ids)):
        text_ids = batch[input_sort_ids[i]]
        text_ids_pad[i, :text_ids.size(0)] = text_ids

    return text_ids_pad, input_lens_sorted, input_sort_ids.argsort()


class FastPitch(_FastPitch):
    def __init__(self,
                 checkpoint: str = None,         
                 arabic_in: bool = True,
                 device=None,
                 **kwargs):
        from models.fastpitch import net_config
        sds = torch.load(checkpoint)
        if 'config' in sds:
            net_config = sds['config']
        super().__init__(**net_config)
        #self.n_eos = len(EOS_TOKENS)
        self.arabic_in = arabic_in

        #if checkpoint is not None:            
        self.load_state_dict(sds['model'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device is None else device

        self.eval()

    def cuda(self):        
        self.device = torch.device('cuda')
        return super().cuda()

    def cpu(self):        
        self.device = torch.device('cpu')
        return super().cpu()

    def to(self, device=None, **kwargs):        
        self.device = device
        return super().to(device=device, **kwargs)

    def _tokenize(self, utterance: str):
        if self.arabic_in:
            return text.arabic_to_tokens(utterance, append_space=False)
        return text.buckwalter_to_tokens(utterance, append_space=False)

    @torch.inference_mode()
    def ttmel_single(self,
                     utterance: str,
                     speed: float = 1,
                     speaker_id: int = 0):

        tokens = self._tokenize(utterance)

        token_ids = text.tokens_to_ids(tokens)
        ids_batch = torch.LongTensor(token_ids).unsqueeze(0).to(self.device)
        sid = torch.LongTensor([speaker_id]).to(self.device)

        # Infer spectrogram and wave      
        (mel_spec, dec_lens, dur_pred, 
            pitch_pred, energy_pred) = self.infer(ids_batch, pace=speed)

        mel_spec = mel_spec[0]

        return mel_spec  # [F, T]

    @torch.inference_mode()
    def ttmel_batch(self,
                    batch: List[str],
                    speed: float = 1,
                    speaker_id: int = 0
                    ):

        batch_tokens = [self._tokenize(line) for line in batch]

        batch_ids = [torch.LongTensor(text.tokens_to_ids(tokens))
                     for tokens in batch_tokens]

        batch = text_collate_fn(batch_ids)
        (
            batch_ids_padded, batch_lens_sorted,
            reverse_sort_ids
        ) = batch

        batch_ids_padded = batch_ids_padded.to(self.device)
        batch_lens_sorted = batch_lens_sorted.to(self.device)

        batch_sids = batch_lens_sorted*0 + speaker_id

        y_pred = self.infer(batch_ids_padded, pace=speed)     
        mel_outputs, mel_specgram_lengths, *_ = y_pred

        mel_list = []
        for i, id in enumerate(reverse_sort_ids):
            mel = mel_outputs[id, :, :mel_specgram_lengths[id]]
            mel_list.append(mel)

        return mel_list

    def ttmel(self,
              text_buckw: Union[str, List[str]],
              speed: float = 1,
              speaker_id: int = 0,
              batch_size: int = 1,
              ):
        # input: string
        if isinstance(text_buckw, str):
            return self.ttmel_single(text_buckw, speed=speed, speaker_id=speaker_id)

        # input: list
        assert isinstance(text_buckw, list)
        batch = text_buckw
        mel_list = []

        if batch_size == 1:
            for sample in batch:
                mel = self.ttmel_single(sample, speed=speed, speaker_id=speaker_id)
                mel_list.append(mel)
            return mel_list

        # infer one batch
        if len(batch) <= batch_size:
            return self.ttmel_batch(batch, speed=speed, speaker_id=speaker_id)

        # batched inference
        batches = [batch[k:k+batch_size]
                   for k in range(0, len(batch), batch_size)]

        for batch in batches:
            mels = self.ttmel_batch(batch, speed=speed, speaker_id=speaker_id)
            mel_list += mels

        return mel_list


class FastPitch2Wave(nn.Module):
    def __init__(self,
                 model_sd_path,
                 vocoder_sd=None,
                 vocoder_config=None,
                 arabic_in: bool = True
                 ):

        super().__init__()

        # from models.fastpitch import net_config
        state_dicts = torch.load(model_sd_path)
        # if 'config' in state_dicts:
        #     net_config = state_dicts['config']

        model = FastPitch(model_sd_path, arabic_in=arabic_in)       
        model.load_state_dict(state_dicts['model'])
        self.model = model

        if vocoder_sd is None or vocoder_config is None:
            config = get_basic_config()
            vocoder_sd = config.vocoder_state_path
            vocoder_config = config.vocoder_config_path

        vocoder = load_hifigan(vocoder_sd, vocoder_config)
        self.vocoder = vocoder
        self.denoiser = Denoiser(vocoder)

        self.eval()

    def forward(self, x):
        return x

    @torch.inference_mode()
    def tts_single(self,
                   text_buckw: str,
                   speed: float = 1,  
                   speaker_id: int = 0,
                   denoise: float = 0,                               
                   return_mel=False):

        mel_spec = self.model.ttmel_single(text_buckw, speed, speaker_id)
          
        wave = self.vocoder(mel_spec)

        if denoise > 0:
            wave = self.denoiser(wave, denoise)

        if return_mel:
            return wave[0].cpu(), mel_spec

        return wave[0].cpu()

    @torch.inference_mode()
    def tts_batch(self,
                  batch: List[str],
                  speed: float = 1,  
                  speaker_id: int = 0,   
                  denoise: float = 0,                             
                  return_mel=False):

        mel_list = self.model.ttmel_batch(batch, speed, speaker_id)

        wav_list = []
        for mel in mel_list:  
            wav_inferred = self.vocoder(mel)
            if denoise > 0:
                wav_inferred = self.denoiser(wav_inferred, denoise)

            wav_list.append(wav_inferred[0].cpu())

        if return_mel:
            wav_list, mel_list

        return wav_list

    def tts(self,
            text_buckw: Union[str, List[str]],
            speed: float = 1,   
            denoise: float = 0, 
            speaker_id: int = 0,
            batch_size: int = 2,                
            return_mel: bool = False):

        # input: string
        if isinstance(text_buckw, str):
            return self.tts_single(text_buckw, speaker_id=speaker_id, 
                                   speed=speed, denoise=denoise,                                
                                   return_mel=return_mel)

        # input: list
        assert isinstance(text_buckw, list)
        batch = text_buckw
        wav_list = []

        if batch_size == 1:
            for sample in batch:
                wav = self.tts_single(sample, speaker_id=speaker_id,
                                      speed=speed, denoise=denoise,                               
                                      return_mel=return_mel)
                wav_list.append(wav)
            return wav_list

        # infer one batch
        if len(batch) <= batch_size:
            return self.tts_batch(batch, speaker_id=speaker_id,
                                  speed=speed, denoise=denoise,                       
                                  return_mel=return_mel)

        # batched inference
        batches = [batch[k:k+batch_size]
                   for k in range(0, len(batch), batch_size)]

        for batch in batches:
            wavs = self.tts_batch(batch, speaker_id=speaker_id,
                                  speed=speed, denoise=denoise,                            
                                  return_mel=return_mel)
            wav_list += wavs

        return wav_list