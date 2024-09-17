from __future__ import annotations

from typing import Any, Dict, Tuple, Union, Optional

import torch
import yaml
from huggingface_hub import hf_hub_download
from torch import nn
from . import config_22k, config_24k
from .feature_extractors import FeatureExtractor, MelSpectrogramFeatures
from .heads import FourierHead, ISTFTHead
from .models import Backbone, VocosBackbone


def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


class MelVocos(nn.Module):
    def __init__(self, config_name='24k'):
        super().__init__()
        
        config = {'22k': config_22k, '24k': config_24k,}[config_name]
        
        self.feature_extractor = MelSpectrogramFeatures(
        **config['feature_extractor']['init_args'])
        self.backbone = VocosBackbone(**config['backbone']['init_args'])
        self.head = ISTFTHead(**config['head']['init_args'])
        
        self.n_mels = config['feature_extractor']['init_args']['n_mels']
        
        
        self.bias_vec = self.make_denoising_vector()
        def new_denoising_vector(m, y):
            m.bias_vec = m.make_denoising_vector()        
        self.register_load_state_dict_post_hook(new_denoising_vector) 
           
        # self.register_buffer('bias_vec', self.make_denoising_vector())                
    
    @property
    def device(self):
        return next(self.parameters()).device    
    
    @torch.inference_mode()
    def make_denoising_vector(self):
        mel_rand = torch.zeros((1, self.n_mels, 88), device=self.device)
        bias_feats = self.backbone(mel_rand)

        x_bias = self.head.out(bias_feats).transpose(1, 2)
        mag_bias, _ = x_bias.chunk(2, dim=1)
        mag_bias = torch.exp(mag_bias)
        mag_bias = torch.clip(mag_bias, max=1e2)  # safeguard to prevent excessively large magnitudes
        
        mag_bias_vec = mag_bias[:,:,0:1] # [1, 513, 1]
        
        return mag_bias_vec        
    
    def forward(self, mel_spec, denoise=0.): # [B, bands=100, frames]           
        
        bb_feats = self.backbone(mel_spec)
        x = self.head.out(bb_feats).transpose(1, 2)
        
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        # mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x, y = torch.cos(p), torch.sin(p)
        # phase = torch.atan2(y, x)
        
        mag = mag - denoise*self.bias_vec #.to(mel_spec.device)
        
        mag = torch.clamp(mag, min=0., max=1e2)
        
        S = mag * (x + 1j * y)

        wave = self.head.istft(S)
        
        return wave # [B, samples]
    
    def reconstruct(self, wave, denoise=0.):
        mel_spec = self.feature_extractor(wave)
        return self.forward(mel_spec, denoise=denoise)


class Vocos(nn.Module):
    """
    The Vocos class represents a Fourier-based neural vocoder for audio synthesis.
    This class is primarily designed for inference, with support for loading from pretrained
    model checkpoints. It consists of three main components: a feature extractor,
    a backbone, and a head.
    """

    def __init__(
        self, feature_extractor: FeatureExtractor, backbone: Backbone, head: FourierHead,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_hparams(cls, config_path: str) -> Vocos:
        """
        Class method to create a new Vocos model instance from hyperparameters stored in a yaml configuration file.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        feature_extractor = instantiate_class(args=(), init=config["feature_extractor"])
        backbone = instantiate_class(args=(), init=config["backbone"])
        head = instantiate_class(args=(), init=config["head"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: Optional[str] = None) -> Vocos:
        """
        Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging Face model hub.
        """
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml", revision=revision)
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", revision=revision)
        model = cls.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu")
        # if isinstance(model.feature_extractor, EncodecFeatures):
        #     encodec_parameters = {
        #         "feature_extractor.encodec." + key: value
        #         for key, value in model.feature_extractor.encodec.state_dict().items()
        #     }
        #     state_dict.update(encodec_parameters)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.inference_mode()
    def forward(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to run a copy-synthesis from audio waveform. The feature extractor first processes the audio input,
        which is then passed through the backbone and the head to reconstruct the audio output.

        Args:
            audio_input (Tensor): The input tensor representing the audio waveform of shape (B, T),
                                        where B is the batch size and L is the waveform length.


        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        features = self.feature_extractor(audio_input, **kwargs)
        audio_output = self.decode(features, **kwargs)
        return audio_output

    @torch.inference_mode()
    def decode(self, features_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to decode audio waveform from already calculated features. The features input is passed through
        the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output

    # @torch.inference_mode()
    # def codes_to_features(self, codes: torch.Tensor) -> torch.Tensor:
    #     """
    #     Transforms an input sequence of discrete tokens (codes) into feature embeddings using the feature extractor's
    #     codebook weights.

    #     Args:
    #         codes (Tensor): The input tensor. Expected shape is (K, L) or (K, B, L),
    #                         where K is the number of codebooks, B is the batch size and L is the sequence length.

    #     Returns:
    #         Tensor: Features of shape (B, C, L), where B is the batch size, C denotes the feature dimension,
    #                 and L is the sequence length.
    #     """
    #     assert isinstance(
    #         self.feature_extractor, EncodecFeatures
    #     ), "Feature extractor should be an instance of EncodecFeatures"

    #     if codes.dim() == 2:
    #         codes = codes.unsqueeze(1)

    #     n_bins = self.feature_extractor.encodec.quantizer.bins
    #     offsets = torch.arange(0, n_bins * len(codes), n_bins, device=codes.device)
    #     embeddings_idxs = codes + offsets.view(-1, 1, 1)
    #     features = torch.nn.functional.embedding(embeddings_idxs, self.feature_extractor.codebook_weights).sum(dim=0)
    #     features = features.transpose(1, 2)

    #     return features
