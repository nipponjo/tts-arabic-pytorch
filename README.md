# tts-arabic-pytorch
[[Samples 1]](https://nipponjo.github.io/tts-arabic-samples) 
[[Samples 2]](https://nipponjo.github.io/tts-arabic-speakers) 
[[Package / ONNX models]](https://github.com/nipponjo/tts_arabic) 
[[Flutter app]](https://github.com/nipponjo/tts-arabic-flutter)
[[arXiv]](https://arxiv.org/abs/2512.00937)

TTS models (Tacotron2, FastPitch), trained on [Nawar Halabi](https://github.com/nawarhalabi)'s [Arabic Speech Corpus](http://en.arabicspeechcorpus.com/), including the [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan) for direct TTS inference.

<div align="center">
  <img src="https://user-images.githubusercontent.com/28433296/227660976-0d1e2033-276e-45e5-b232-a5a9b6b3f2a8.png" width="95%"></img>
</div>

**Manuscript**

More information about this repository can also be found in the manuscript *Arabic TTS with FastPitch: Reproducible Baselines, Adversarial Training, and Oversmoothing Analysis* ([arXiv](https://arxiv.org/abs/2512.00937) | [ResearchGate](https://www.researchgate.net/publication/398225096_Arabic_TTS_with_FastPitch_Reproducible_Baselines_Adversarial_Training_and_Oversmoothing_Analysis)).

**Related Papers**

Tacotron2 | *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions* ([arXiv](https://arxiv.org/abs/1712.05884))

FastPitch | *FastPitch: Parallel Text-to-speech with Pitch Prediction* ([arXiv](https://arxiv.org/abs/2006.06873))

HiFi-GAN  | *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis* ([arXiv](https://arxiv.org/abs/2010.05646))



## Audio Samples

You can listen to some audio samples [here](https://nipponjo.github.io/tts-arabic-samples).

## Multispeaker model (in progress)

Multispeaker weights are available for the FastPitch model.
Currently, another male voice and two female voices have been added.
Audio samples can be found [here](https://nipponjo.github.io/tts-arabic-speakers). Download weights [here](https://drive.google.com/u/0/uc?id=18IYUSRXvLErVjaDORj_TKzUxs90l61Ja&export=download). There also exists an [ONNX version](https://github.com/nipponjo/tts_arabic) for this model. 

The multispeaker dataset was created by synthesizing data with [Coqui](https://github.com/coqui-ai)'s [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) model and a mix of voices from the [Tunisian_MSA](https://www.openslr.org/46/) dataset.

## Quick Setup

The models were trained with the mse loss as described in the papers. I also trained the models using an additional adversarial loss (adv). The difference is not large, but I think that the (adv) version often sounds a bit clearer. You can compare them yourself.

Running `python download_files.py` will download all pretrained weights, alternatively:

Download the pretrained weights for the Tacotron2 model ([mse](https://drive.google.com/u/0/uc?id=1GCu-ZAcfJuT5qfzlKItcNqtuVNa7CNy9&export=download) | [adv](https://drive.google.com/u/0/uc?id=1FusCFZIXSVCQ9Q6PLb91GIkEnhn_zWRS&export=download)).

Download the pretrained weights for the FastPitch model ([mse](https://drive.google.com/u/0/uc?id=1sliRc62wjPTnPWBVQ95NDUgnCSH5E8M0&export=download) | [adv](https://drive.google.com/u/0/uc?id=1-vZOhi9To_78-yRslC6sFLJBUjwgJT-D&export=download)).

Download the [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan) weights ([link](https://drive.google.com/u/0/uc?id=1zSYYnJFS-gQox-IeI71hVY-fdPysxuFK&export=download)). Either put them into `pretrained/hifigan-asc-v1` or edit the following lines in `configs/basic.yaml`.

```yaml
# vocoder
vocoder_state_path: pretrained/hifigan-asc-v1/hifigan-asc.pth
vocoder_config_path: pretrained/hifigan-asc-v1/config.json
```

This repo includes the diacritization models [Shakkala](https://github.com/Barqawiz/Shakkala) and [Shakkelha](https://github.com/AliOsm/shakkelha). 

The weights can be downloaded [here](https://drive.google.com/u/1/uc?id=1MIZ_t7pqAQP-R3vwWWQTJMER8yPm1uB1&export=download). There also exists a [separate repo](https://github.com/nipponjo/arabic-vocalization) and [package](https://github.com/nipponjo/arabic_vocalizer).

-> Alternatively, [download all models](https://drive.google.com/u/1/uc?id=1FD2J-xUk48JPF9TeS8ZKHzDC_ZNBfLd8&export=download) and put the content of the zip file into the `pretrained` folder.

## Required packages:

`torch torchaudio pyyaml`

~ for training: `librosa matplotlib tensorboard`

~ for the demo app: `fastapi "uvicorn[standard]"`

## Using the models

The `Tacotron2`/`FastPitch` from `models.tacotron2`/`models.fastpitch` are wrappers that simplify text-to-mel inference. The `Tacotron2Wave`/`FastPitch2Wave` models includes the [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan) for direct text-to-speech inference.

## Inference options

```python
text = "اَلسَّلامُ عَلَيكُم يَا صَدِيقِي."

wave = model.tts(
    text_input = text, # input text
    speed = 1, # speaking speed
    denoise = 0.005, # HifiGAN denoiser strength
    speaker_id = 0, # speaker id
    batch_size = 2, # batch size for batched inference
    vowelizer = None, # vowelizer model
    pitch_mul = 1, # pitch multiplier (for FastPitch)
    pitch_add = 0, # pitch offset (for FastPitch)
    return_mel = False # return mel spectrogram?
)
```

## Inferring the Mel spectrogram

```python
from models.tacotron2 import Tacotron2
model = Tacotron2('pretrained/tacotron2_ar_adv.pth')
model = model.cuda()
mel_spec = model.ttmel("اَلسَّلامُ عَلَيكُم يَا صَدِيقِي.")
```

```python
from models.fastpitch import FastPitch
model = FastPitch('pretrained/fastpitch_ar_adv.pth')
model = model.cuda()
mel_spec = model.ttmel("اَلسَّلامُ عَلَيكُم يَا صَدِيقِي.")
```

## End-to-end Text-to-Speech

```python
from models.tacotron2 import Tacotron2Wave
model = Tacotron2Wave('pretrained/tacotron2_ar_adv.pth')
model = model.cuda()
wave = model.tts("اَلسَّلامُ عَلَيكُم يَا صَدِيقِي.")

wave_list = model.tts(["صِفر" ,"واحِد" ,"إِثنان", "ثَلاثَة" ,"أَربَعَة" ,"خَمسَة", "سِتَّة" ,"سَبعَة" ,"ثَمانِيَة", "تِسعَة" ,"عَشَرَة"])
```

```python
from models.fastpitch import FastPitch2Wave
model = FastPitch2Wave('pretrained/fastpitch_ar_adv.pth')
model = model.cuda()
wave = model.tts("اَلسَّلامُ عَلَيكُم يَا صَدِيقِي.")

wave_list = model.tts(["صِفر" ,"واحِد" ,"إِثنان", "ثَلاثَة" ,"أَربَعَة" ,"خَمسَة", "سِتَّة" ,"سَبعَة" ,"ثَمانِيَة", "تِسعَة" ,"عَشَرَة"])
```

By default, Arabic letters are converted using the [Buckwalter transliteration](https://en.wikipedia.org/wiki/Buckwalter_transliteration), which can also be used directly.

```python
wave = model.tts(">als~alAmu Ealaykum yA Sadiyqiy.")
wave_list = model.tts(["Sifr", "wAHid", "<i^nAn", "^alA^ap", ">arbaEap", "xamsap", "sit~ap", "sabEap", "^amAniyap", "tisEap", "Ea$arap"])
```

## Unvocalized text

```python
text_unvoc = "اللغة العربية هي أكثر اللغات السامية تحدثا، وإحدى أكثر اللغات انتشارا في العالم"
wave_shakkala = model.tts(text_unvoc, vowelizer='shakkala')
wave_shakkelha = model.tts(text_unvoc, vowelizer='shakkelha')
```


### Inference from text file

```bash
python inference.py
# default parameters:
python inference.py --list data/infer_text.txt --out_dir samples/results --model fastpitch --checkpoint pretrained/fastpitch_ar_adv.pth --batch_size 2 --denoise 0
```

## Testing the model

To test the model run:
```bash
python test.py
# default parameters:
python test.py --model fastpitch --checkpoint pretrained/fastpitch_ar_adv.pth --out_dir samples/test
```

## Processing details

This repo uses Nawar Halabi's [Arabic-Phonetiser](https://github.com/nawarhalabi/Arabic-Phonetiser) but simplifies the result such that different contexts are ignored (see `text/symbols.py`). Further, a doubled consonant is represented as consonant + doubling-token.

The Tacotron2 model can sometimes struggle to pronounce the last phoneme of a sentence when it ends in an unvocalized consonant. The pronunciation is more reliable if one appends a word-separator token at the end and cuts it off using the alignments weights (details in `models.networks`). This option is implemented as a default postprocessing step that can be disabled by setting `postprocess_mel=False`.


## Training the model

Before training, the audio files must be resampled. The model was trained after preprocessing the files using `scripts/preprocess_audio.py`.

To train the model with options specified in the config file run:
```bash
python train.py
# default parameters:
python train.py --config configs/nawar.yaml
```


## Web app

The web app uses the FastAPI library. To run the app you need the following packages:

fastapi: for the backend api | uvicorn: for serving the app

Install with: `pip install fastapi "uvicorn[standard]"`

Run with: `python app.py`

Preview:

<div align="center">
  <img src="https://user-images.githubusercontent.com/28433296/212092260-57b2ced3-da69-48ad-8be7-50e621423687.png" width="66%"></img>
</div>



## Acknowledgements

I referred to NVIDIA's [Tacotron2 implementation](https://github.com/NVIDIA/tacotron2) for details on model training. 

The FastPitch files stem from NVIDIA's [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/)
