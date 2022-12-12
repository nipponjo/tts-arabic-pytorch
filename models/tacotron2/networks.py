from typing import List, Union

import text
import torch
import torch.nn as nn
from .tacotron2_ms import Tacotron2MS

from text.symbols import EOS_TOKENS, SEPARATOR_TOKEN
from utils import get_basic_config
from vocoder import load_hifigan
from vocoder.hifigan.denoiser import Denoiser


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


def needs_postprocessing(token: str):
    return token not in [
        'a', 'i', 'u', 'aa', 'ii', 'uu', 'n', 'm', 'h']


def truncate_mel(mel_spec: torch.Tensor, ps_end):
    ps_end_max = ps_end.max()
    n_end = next(i for i in range(len(ps_end)) if ps_end[i] >= 0.8*ps_end_max)
    mel_cut = mel_spec[:, :n_end]
    mel_cut = torch.nn.functional.pad(mel_cut, (0, 3), mode='replicate')
    return mel_cut


def resize_mel(mel: torch.Tensor,
               rate: Union[int, float] = 1.0,
               mode: str = 'bicubic'):
    """
    Args:
        mel: mel spectrogram [num_mels, spec_length]
    Returns:
        resized_mel [num_mels, new_spec_length]
    """
    Nf, Nt = mel.shape[-2:]
    Nt_new = int(1 / rate * Nt)
    if Nt == Nt_new:
        return mel
    mel_res = torch.nn.functional.interpolate(mel[None, None, ...],
                                              (Nf, Nt_new), mode=mode)[0, 0]
    return mel_res



class Tacotron2(Tacotron2MS):
    def __init__(self,
                 checkpoint: str = None,
                 n_symbol: int = 40,
                 decoder_max_step: int = 3000,
                 arabic_in: bool = True,
                 device=None,
                 **kwargs):
        super().__init__(n_symbol=n_symbol,
                         decoder_max_step=decoder_max_step,
                         **kwargs)
        self.n_eos = len(EOS_TOKENS)
        self.arabic_in = arabic_in

        if checkpoint is not None:
            sds = torch.load(checkpoint)
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
            return text.arabic_to_tokens(utterance)
        return text.buckwalter_to_tokens(utterance)

    @torch.inference_mode()
    def ttmel_single(self,
                     utterance: str,                  
                     speaker_id: int = 0,
                     speed: Union[int, float, None] = None,
                     postprocess_mel: bool = True):

        tokens = self._tokenize(utterance)

        process_mel = False
        if postprocess_mel and needs_postprocessing(tokens[-self.n_eos-1]):
            tokens.insert(-self.n_eos, SEPARATOR_TOKEN)
            process_mel = True

        token_ids = text.tokens_to_ids(tokens)
        ids_batch = torch.LongTensor(token_ids).unsqueeze(0).to(self.device)
        sid = torch.LongTensor([speaker_id]).to(self.device)

        # Infer spectrogram and wave
        mel_spec, _, alignments = self.infer(ids_batch, sid)
        mel_spec = mel_spec[0]
        if process_mel:
            mel_spec = truncate_mel(mel_spec, alignments[0, :, -self.n_eos-1])

        if speed is not None:
            mel_spec = resize_mel(mel_spec, rate=speed)

        return mel_spec  # [F, T]

    @torch.inference_mode()
    def ttmel_batch(self,
                    batch: List[str],
                    speaker_id: int = 0,
                    speed: Union[int, float, None] = None,
                    postprocess_mel: bool = True):

        batch_tokens = [self._tokenize(line) for line in batch]

        list_postprocess = []
        if postprocess_mel:
            for i in range(len(batch_tokens)):
                process_mel = False
                if needs_postprocessing(batch_tokens[i][-self.n_eos-1]):
                    batch_tokens[i].insert(-self.n_eos, SEPARATOR_TOKEN)
                    process_mel = True
                list_postprocess.append(process_mel)

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

        y_pred = self.infer(batch_ids_padded, batch_sids, batch_lens_sorted)
        mel_outputs_postnet, mel_specgram_lengths, alignments = y_pred

        mel_list = []
        for i, id in enumerate(reverse_sort_ids):

            mel = mel_outputs_postnet[id, :, :mel_specgram_lengths[id]]

            if postprocess_mel and list_postprocess[i]:
                ps_end = alignments[id,
                                    :mel_specgram_lengths[id],
                                    batch_lens_sorted[id]-self.n_eos-1]
                mel = truncate_mel(mel, ps_end)

            if speed is not None:
                mel = resize_mel(mel, rate=speed)

            mel_list.append(mel)

        return mel_list

    def ttmel(self,
              text_buckw: Union[str, List[str]],
              speaker_id: int = 0,
              speed: Union[int, float, None] = None,
              batch_size: int = 8,
              postprocess_mel: bool = True):
        # input: string
        if isinstance(text_buckw, str):
            return self.ttmel_single(text_buckw, speaker_id, speed, postprocess_mel)

        # input: list
        assert isinstance(text_buckw, list)
        batch = text_buckw
        mel_list = []

        if batch_size == 1:
            for sample in batch:
                mel = self.ttmel_single(sample, speaker_id, speed, postprocess_mel)
                mel_list.append(mel)
            return mel_list

        # infer one batch
        if len(batch) <= batch_size:
            return self.ttmel_batch(batch, speaker_id, speed, postprocess_mel)

        # batched inference
        batches = [batch[k:k+batch_size]
                   for k in range(0, len(batch), batch_size)]

        for batch in batches:
            mels = self.ttmel_batch(batch, speaker_id, speed, postprocess_mel)   
            mel_list += mels

        return mel_list


class Tacotron2Wave(nn.Module):
    def __init__(self,
                 model_sd_path,
                 vocoder_sd=None,
                 vocoder_config=None,
                 arabic_in: bool = True,
                 n_symbol: int = 40):

        super().__init__()

        model = Tacotron2(n_symbol=n_symbol, arabic_in=arabic_in)

        state_dicts = torch.load(model_sd_path)
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
                   speed: Union[int, float, None] = None,
                   speaker_id: int = 0,
                   denoise: float = 0,
                   postprocess_mel=True,
                   return_mel=False):

        mel_spec = self.model.ttmel_single(text_buckw, speaker_id, speed, postprocess_mel)
        # if speed is not None:
        #     mel_spec = resize_mel(mel_spec, rate=speed)

        wave = self.vocoder(mel_spec)

        if denoise > 0:
            wave = self.denoiser(wave, denoise)

        if return_mel:
            return wave[0].cpu(), mel_spec

        return wave[0].cpu()

    @torch.inference_mode()
    def tts_batch(self,
                  batch: List[str],
                  speed: Union[int, float, None] = None,
                  denoise: float = 0,
                  speaker_id: int = 0,                  
                  postprocess_mel=True,
                  return_mel=False):

        mel_list = self.model.ttmel_batch(batch, speaker_id, speed, postprocess_mel)

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
            speed: Union[int, float, None] = None,
            denoise: float = 0,
            speaker_id: int = 0,
            batch_size: int = 8,
            postprocess_mel: bool = True,
            return_mel: bool = False):

        # input: string
        if isinstance(text_buckw, str):
            return self.tts_single(text_buckw, speaker_id=speaker_id, 
                                   speed=speed, denoise=denoise,
                                   postprocess_mel=postprocess_mel,
                                   return_mel=return_mel)

        # input: list
        assert isinstance(text_buckw, list)
        batch = text_buckw
        wav_list = []

        if batch_size == 1:
            for sample in batch:
                wav = self.tts_single(sample, speaker_id=speaker_id,
                                      speed=speed, denoise=denoise,
                                      postprocess_mel=postprocess_mel,
                                      return_mel=return_mel)
                wav_list.append(wav)
            return wav_list

        # infer one batch
        if len(batch) <= batch_size:
            return self.tts_batch(batch, speaker_id=speaker_id,
                                  speed=speed, denoise=denoise,
                                  postprocess_mel=postprocess_mel,
                                  return_mel=return_mel)

        # batched inference
        batches = [batch[k:k+batch_size]
                   for k in range(0, len(batch), batch_size)]

        for batch in batches:
            wavs = self.tts_batch(batch,  speaker_id=speaker_id,
                                  speed=speed, denoise=denoise,
                                  postprocess_mel=postprocess_mel,
                                  return_mel=return_mel)
            wav_list += wavs

        return wav_list
