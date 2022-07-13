import os
import torch
import torchaudio

import text
from model.networks import Tacotron2Wave


class TTSManager:
    def __init__(self, checkpoint, out_dir):

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f"Created folder: {out_dir}")

        device = torch.device('cuda'
                              if torch.cuda.is_available() else 'cpu')

        # Load tts model
        model = Tacotron2Wave(checkpoint, arabic_in=False)
        model.to(device)

        self.sample_rate = 22_050
        self.model = model
        self.out_dir = out_dir

    def tts(self, text_buckw, speed=1):
        # Process utterance
        postprocess = True
        if text_buckw.endswith('#'):
            postprocess = False
        wave = self.model.tts(text_buckw, speed=speed,
                              postprocess_mel=postprocess)

        torchaudio.save(f'{self.out_dir}/wave.wav',
                        wave.unsqueeze(0).cpu(), self.sample_rate)

        phonemes = text.buckwalter_to_phonemes(text_buckw)
        phonemes = text.simplify_phonemes(
            phonemes.replace(" ", "").replace("+", " "))
        response_data = {'phonemes': phonemes}

        return response_data
