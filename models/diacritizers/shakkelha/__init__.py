import torch
from .symbols import (RNN_BIG_CHARACTERS_MAPPING, 
DIACRITICS_LIST, ARABIC_LETTERS_LIST, RNN_REV_CLASSES_MAPPING, RNN_SMALL_CHARACTERS_MAPPING)


def remove_diacritics(data, DIACRITICS_LIST):
  return data.translate(str.maketrans('', '', ''.join(DIACRITICS_LIST)))

CHARACTERS_MAPPING = RNN_BIG_CHARACTERS_MAPPING
# CHARACTERS_MAPPING = RNN_SMALL_CHARACTERS_MAPPING
REV_CLASSES_MAPPING = RNN_REV_CLASSES_MAPPING


def encode(input_text:str):
    x = [CHARACTERS_MAPPING['<SOS>']]
    for idx, char in enumerate(input_text):
        if char in DIACRITICS_LIST:
            continue
        if char not in CHARACTERS_MAPPING:
            x.append(CHARACTERS_MAPPING['<UNK>'])
        else:
            x.append(CHARACTERS_MAPPING[char])
            
    x.append(CHARACTERS_MAPPING['<EOS>'])

    return x

def decode(probs, input_text:str):
    probs = probs[0][1:]

    output = ''
    for char, prediction in zip(remove_diacritics(input_text, DIACRITICS_LIST), probs):
        output += char

        if char not in ARABIC_LETTERS_LIST:
            continue

        prediction = torch.argmax(prediction).item()

        if '<' in REV_CLASSES_MAPPING[prediction]:
            continue

        output += REV_CLASSES_MAPPING[prediction]
    
    return output