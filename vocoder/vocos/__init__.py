# from vocos.pretrained import Vocos


__version__ = "0.1.0"



config_24k = {'sample_rate': 24000,
 'initial_learning_rate': '5e-4',
 'mel_loss_coeff': 45,
 'mrd_loss_coeff': 0.1,
 'num_warmup_steps': 0,
 'pretrain_mel_steps': 0,
 'evaluate_utmos': True,
 'evaluate_pesq': True,
 'evaluate_periodicty': True,
 'feature_extractor': {'class_path': 'vocos.feature_extractors.MelSpectrogramFeatures',
  'init_args': {'sample_rate': 24000,
   'n_fft': 1024,
   'hop_length': 256,
   'n_mels': 100,
   'padding': 'center'}},
 'backbone': {'class_path': 'vocos.models.VocosBackbone',
  'init_args': {'input_channels': 100,
   'dim': 512,
   'intermediate_dim': 1536,
   'num_layers': 8}},
 'head': {'class_path': 'vocos.heads.ISTFTHead',
  'init_args': {'dim': 512,
   'n_fft': 1024,
   'hop_length': 256,
   'padding': 'center'}}}


config_22k = {'sample_rate': 22050,
 'initial_learning_rate': '5e-4',
 'mel_loss_coeff': 45,
 'mrd_loss_coeff': 0.1,
 'num_warmup_steps': 0,
 'pretrain_mel_steps': 0,
 'evaluate_utmos': True,
 'evaluate_pesq': True,
 'evaluate_periodicty': True,
 'feature_extractor': {'class_path': 'vocos.feature_extractors.MelSpectrogramFeatures',
  'init_args': {
   'sample_rate': 24000,
   'n_fft': 1024,
   'hop_length': 256,   
   'n_mels': 80,
   
    'padding': 'same',
    'f_min': 0,
    'f_max': 8000,
    'norm': "slaney",
    'mel_scale': "slaney",
   
   }},
 'backbone': {'class_path': 'vocos.models.VocosBackbone',
  'init_args': {'input_channels': 80,
   'dim': 512,
   'intermediate_dim': 1536,
   'num_layers': 8}},
 'head': {'class_path': 'vocos.heads.ISTFTHead',
  'init_args': {'dim': 512,
   'n_fft': 1024,
   'hop_length': 256,
   'padding': 'same'}}}
