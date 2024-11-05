# Model training

## Background

Text (&rarr; Phonemes) &rarr; Token ids &rarr; Mel frames &rarr; Audio wave

Text (&rarr; Phonemizer) &rarr; Tokenizer &rarr; TTS (Mel output) &rarr; Vocoder &rarr; Audio wave

A *tokenizer* maps the text or phonetic symbols to token ids. In this repo, this is implemented in the `/text` folder.
A *phonemizer* can be helpful when pronunciation rules are difficult or to allow phoneme based manipulation. For MSA the pronunciation rules for diacritized text are not overly complicated (`text/phonetise_buckwalter.py`) such that a model can learn these from a sufficiently large dataset. 
A *vowelizer* or *diacritizer* can try to estimate the diacrits but these are not perfect and only work for MSA.

It is common to have two models, one that performs the *Token ids &rarr; Mel frames mapping* such as [Tacotron2](https://arxiv.org/abs/1712.05884) or [FastPitch](https://arxiv.org/abs/2006.06873) and a *vocoder* such as [Hifi-GAN](https://arxiv.org/abs/2010.05646) or [Vocos](https://arxiv.org/abs/2306.00814) that performs the *Mel frames &rarr; Audio wave* mapping. The advantage of this approach is that learning to map tokens to mel frames is easier / faster and the vocoder can be trained in a self-supervised manner. Different *Token &rarr; Mel* models can be combined with the same vocoder which offers some flexibility.
A one-stage model like [VITS](https://arxiv.org/abs/2106.06103) merges these two stages and therefore avoids the mel spectrogram bottleneck but usually takes much longer to train, as the the adversarial losses need to do a lot of work (as for the vocoder), and may need a larger dataset. It may depend on the use case which approach is preferred.


## Config file

Example config files for the [Arabic Speech Corpus](https://en.arabicspeechcorpus.com/) are given in the `/configs` folder. 
See `nawar_fp_adv_raw.yaml` for an example for training on raw input.

Here it is assumed that we have audio files like:
```
I:/tts/3arabiyya/arabic-speech-corpus/wav_red/ARA NORM  0002.wav
I:/tts/3arabiyya/arabic-speech-corpus/wav_red/ARA NORM  0003.wav
I:/tts/3arabiyya/arabic-speech-corpus/wav_red/ARA NORM  0004.wav
...
```
The folder that contains the audio files is set in `train_wavs_path`.

The file set in `train_labels` should list the audio filenames and the corresponding text. Different ways of providing the filename and text are supported by modifying the `label_pattern`.

The `label_pattern` is a [Regex pattern](https://docs.python.org/3/howto/regex.html#non-capturing-and-named-groups) that is used in the function `_process_line` in `/utils/data.py` to extract the filename and text from each line.

To train the model on raw input letters, use `<raw>` as in:
`label_pattern: '"(?P<filename>.*)" "(?P<raw>.*)"'`

This will process lines as in `/data/train_arab.txt` and map the raw Arabic letters to token ids.

```
"ARA NORM  0002.wav" "وَرَجَّحَ التَّقْرِيرُ الَّذِي أَعَدَّهُ مَعْهَدُ أَبْحَاثِ هَضَبَةِ التِّبِتِ فِي الْأَكَادِيمِيَّةِ الصِّينِيَّةِ لِلْعُلُومِ أَنْ تَسْتَمِرَّ دَرَجَاتُ الْحَرَارَةِ وَمُسْتَوَيَاتُ الرُّطُوبَةِ فِي الْإِرْتِفَاعِ طَوَالَ هَذَا الْقَرْنْ"
"ARA NORM  0003.wav" "مِمَّا قَدْ يُؤَدِّي إِلَى تَرَاجُعِ مَسَاحَاتِ الْأَنْهَارِ الجَّلِيدِيَّةِ وَانْتِشَارِ التَّصَحُّرِ"
"ARA NORM  0004.wav" "وَذَكَرَ التَّقْرِيرُ أَنَ تَرَاجُعَ مَسَاحَةِ الْجَلِيدِ يُمْكِنُ أَيُّخِلَّ بِمُعَدَّلَاتِ إِمْدَادَاتِ الْمِيَاهِ لِعَدَدٍ مِنْ أَنْهَارِ آسْيَا الرَّئِيسِيَّةِ الَّتِي تَمْبُعُ مِنَ الْهَضَبَةِ"
...
```
A `label_pattern: '"(?P<filename>.*)" "(?P<arabic>.*)"'` would phonemize the Arabic text and then map the phonemes to token ids.


For a label file like the following (`data/train_arab2.txt`), the  `label_pattern` should be changed to `'(?P<filename>.*)\|(?P<raw>.*)'` / `'(?P<filename>.*)\|(?P<arabic>.*)'`.

```
ARA NORM  0002.wav|وَرَجَّحَ التَّقْرِيرُ الَّذِي أَعَدَّهُ مَعْهَدُ أَبْحَاثِ هَضَبَةِ التِّبِتِ فِي الْأَكَادِيمِيَّةِ الصِّينِيَّةِ لِلْعُلُومِ أَنْ تَسْتَمِرَّ دَرَجَاتُ الْحَرَارَةِ وَمُسْتَوَيَاتُ الرُّطُوبَةِ فِي الْإِرْتِفَاعِ طَوَالَ هَذَا الْقَرْنْ
ARA NORM  0003.wav|مِمَّا قَدْ يُؤَدِّي إِلَى تَرَاجُعِ مَسَاحَاتِ الْأَنْهَارِ الجَّلِيدِيَّةِ وَانْتِشَارِ التَّصَحُّرِ
ARA NORM  0004.wav|وَذَكَرَ التَّقْرِيرُ أَنَ تَرَاجُعَ مَسَاحَةِ الْجَلِيدِ يُمْكِنُ أَيُّخِلَّ بِمُعَدَّلَاتِ إِمْدَادَاتِ الْمِيَاهِ لِعَدَدٍ مِنْ أَنْهَارِ آسْيَا الرَّئِيسِيَّةِ الَّتِي تَمْبُعُ مِنَ الْهَضَبَةِ
...
```

### Checkpointing

Training is resumed from the checkpoint specified at `restore_model`
`restore_model: ./pretrained/fastpitch_raw_ms.pth`

A checkpoint named `states.pth` is saved every `n_save_states_iter: 100` iterations (batches) at `checkpoint_dir: checkpoints/exp_fp_adv`.

Backup checkpoints named `states_1000.pth, states_2000.pth, ...` are saved every
`n_save_backup_iter: 1000` iterations.

**`restore_model` should be set to the latest checkpoint (e.g, `checkpoints/exp_fp_adv/states.pth`) after training has been resumed from a pretrained checkpoint (like `./pretrained/fastpitch_raw_ms.pth`)** 

## Pitch extraction

The FastPitch model can be conditioned on the speaker's pitch contour. This is not strictly required to train the model but improves the prosody and lets us manipulate the pitch to some extent once the model is trained. This repo contains two scripts for extracting the pitch: `extract_f0_pyin.py` and `extract_f0_penn.py`. 

The first script uses the [pYIN](https://code.soundsoftware.ac.uk/projects/pyin) ([librosa](https://librosa.org/doc/latest/generated/librosa.pyin.html)) algorithm, the second the PENN ([arxiv](https://arxiv.org/abs/2301.12258)|[github](https://github.com/interactiveaudiolab/penn)) neural network model.
In my experience, the pYIN algorithm is a little more accurate but also slow, mostly because it uses the [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm). The PENN model is much faster on GPU and the difference in accuracy is most likely not significant for model conditioning.

The `waves_dir` and `pitches_dir` in the scripts should be changed to your paths.
```python
waves_dir = 'I:/tts/3arabiyya/arabic-speech-corpus/wav_red'
pitches_dir = 'I:/tts/3arabiyya/arabic-speech-corpus/wav_red/pitches'
```

Running the scripts should produce a folder at `pitches_dir` with the pitch files named `ARA NORM  0002.wav.pth, ...`

The pitch directory should be set in the config file:
`f0_folder_path: I:/tts/3arabiyya/arabic-speech-corpus/wav_red/pitches`

The script should also save a text file at `data/mean_std_....txt` with pitch mean and std that are used for normalization and should be set in config file:
`f0_mean: 130.05478`
`f0_std: 22.86267`

## Training / Testing

A checkpoint that has been trained an raw input (the 4 speakers and [Common Voice Arabic](https://commonvoice.mozilla.org/en/datasets)) is `fastpitch_raw_ms.pth` on [Google drive](https://drive.google.com/drive/folders/1Ft2JOt47qNIQzu-Wz9Or5T0nXCc6N9IN?usp=sharing).

Use `train_fp_adv.py` for training. I don't see a reason for training the Tacotron2 model, since the FastPitch model is better in every aspect I am aware of.

`test_raw_model.py` shows how to use the model for inference. With raw input the `phonemize` argument should be set to `False`.

```python
wave = model.tts(text, speaker_id=0, phonemize=False)
```
