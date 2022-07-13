import os

import text
import torch
import torchaudio
from torch.utils.data import Dataset

from utils import read_lines_from_file, progbar
from utils.audio import MelSpectrogram


def text_mel_collate_fn(batch):
    """
    Args:
        batch: List[(text_ids, mel_spec)]
    Returns:
        text_ids_pad
        input_lengths
        mel_pad
        gate_pad
        output_lengths
    """
    input_lens_sorted, input_sort_ids = torch.sort(
        torch.LongTensor([len(x[0]) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lens_sorted[0]

    num_mels = batch[0][1].size(0)
    max_target_len = max([x[1].size(1) for x in batch])

    text_ids_pad = torch.LongTensor(len(batch), max_input_len)
    mel_pad = torch.FloatTensor(len(batch), num_mels, max_target_len)
    gate_pad = torch.FloatTensor(len(batch), max_target_len)
    output_lengths = torch.LongTensor(len(batch))

    text_ids_pad.zero_(), mel_pad.zero_(), gate_pad.zero_()

    for i in range(len(input_sort_ids)):
        text_ids, mel = batch[input_sort_ids[i]]
        text_ids_pad[i, :text_ids.size(0)] = text_ids
        mel_pad[i, :, :mel.size(1)] = mel
        gate_pad[i, mel.size(1)-1:] = 1
        output_lengths[i] = mel.size(1)

    return text_ids_pad, input_lens_sorted, \
        mel_pad, gate_pad, output_lengths


def remove_silence(energy_per_frame: torch.Tensor, thresh: float = -10.0):
    keep = energy_per_frame > thresh
    # keep silence at the end
    i = keep.size(0)-1
    while not keep[i] and i > 0:
        keep[i] = True
        i -= 1
    return keep


class ArabDataset(Dataset):
    def __init__(self, txtpath='./data/train_phon.txt',
                 wavpath='G:/data/arabic-speech-corpus/wav_new',
                 cache=True):
        super().__init__()

        self.mel_fn = MelSpectrogram()
        self.wav_path = wavpath
        self.cache = cache

        lines = read_lines_from_file(txtpath)

        phoneme_mel_list = []

        for line in progbar(lines):
            fname, phonemes = line.split('" "')
            fname, phonemes = fname[1:], phonemes[:-1]

            tokens = text.phonemes_to_tokens(phonemes)
            token_ids = text.tokens_to_ids(tokens)
            fpath = os.path.join(self.wav_path, fname)

            if self.cache:
                mel_log = self._get_mel_from_fpath(fpath)
                phoneme_mel_list.append(
                    (torch.LongTensor(token_ids), mel_log))
            else:
                phoneme_mel_list.append((torch.LongTensor(token_ids), fpath))

        self.data = phoneme_mel_list

    def _get_mel_from_fpath(self, fpath):
        wave, _ = torchaudio.load(fpath)

        mel_raw = self.mel_fn(wave)
        mel_log = mel_raw.clamp_min(1e-5).log().squeeze()

        energy_per_frame = mel_log.mean(0)
        mel_log = mel_log[:, remove_silence(energy_per_frame)]

        return mel_log

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.cache:
            return self.data[idx]

        phonemes, fpath = self.data[idx]
        mel_log = self._get_mel_from_fpath(fpath)

        return phonemes, mel_log
