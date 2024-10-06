# %%

import sounddevice as sd
from models.fastpitch import FastPitch2Wave

def remove_diacrits(text: str): 
    return ''.join([c for c in text if c not in ['َ', 'ِ', 'ُ' , 'ٍ', 'ٌ', 'ْ' , 'ٰ']])

# %%

# https://drive.google.com/file/d/1JcVSKtt6DjGWDsDZwVhoaa3gqCw03L1p/view?usp=sharing
ckpt_path = './pretrained/fastpitch_raw_ms.pth'

model = FastPitch2Wave(ckpt_path).cuda()

# %%

text = "اَلسَّلامُ عَلَيكُم يَا صَدِيقِي."
text = "أَتَاحَتْ لِلْبَائِعِ المُتَجَوِّلِ أنْ يَكُونَ جَاذِباً لِلمُوَاطِنِ الأقَلِّ دَخْلاً."

wave = model.tts(text, speaker_id=0, phonemize=False)

sd.play(wave, 22050)

# %%

text_ = remove_diacrits(text)

wave = model.tts(text_, speaker_id=0, phonemize=False)

sd.play(wave.cpu(), 22050)

