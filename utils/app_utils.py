import os
import torch
import torchaudio

import text

from . import get_custom_config, get_config
from vocoder import load_hifigan
from vocoder.hifigan.denoiser import Denoiser

config = get_config('./configs/basic.yaml')
models_config = get_custom_config('./app/models.yaml')

def load_models():
    models = []
    for model_name, model_dict in models_config.__dict__.items():
        sd_path = model_dict['path']
        if not os.path.exists(sd_path):
            print(f"No model @ {sd_path}")
            continue

        if model_dict['type'] == 'tacotron2':
            from models.tacotron2 import Tacotron2
            model = Tacotron2(sd_path)
        elif model_dict['type'] == 'fastpitch':
            from models.fastpitch import FastPitch
            model = FastPitch(sd_path)
        else:
            print(f"Model type: {model_dict['type']} not supported")
            continue
        model.cuda()
        model.eval()

        models.append((model_name, model))
    
    return models

class TTSManager:
    def __init__(self, out_dir):

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f"Created folder: {out_dir}")

        device = torch.device('cuda'
                              if torch.cuda.is_available() else 'cpu')

        self.vocoder = load_hifigan(config.vocoder_state_path, 
                                    config.vocoder_config_path)
        self.denoiser = Denoiser(self.vocoder, mode='zeros')
        self.vocoder.to(device)
        self.denoiser.to(device)

        self.sample_rate = 22_050
        self.models = load_models()
        self.out_dir = out_dir

    @torch.inference_mode()
    def tts(self, text_buckw, speed=1, denoise=0.01):

        response_data = []

        for i, (model_name, model) in enumerate(self.models):
            model.cuda()
            mel_spec = model.ttmel(text_buckw, speed=speed)
            wave = self.vocoder(mel_spec)
            wave_den = self.denoiser(wave, denoise)

            wave_den /= wave_den.abs().max()
            wave_den *= 0.99

            torchaudio.save(f'./app/static/wave{i}.wav',
                        wave_den.cpu(), 22050)

            response_data.append({
                'name': model_name,
                'phon': '',
                'id': i,
            })        
            model.cpu()


        return response_data
