from text.symbols import symbols, symbols_full, DOUBLING_TOKEN, EOS_TOKEN, SEPARATOR_TOKEN
from text.phonetise_buckwalter import (
    arabic_to_buckwalter,
    buckwalter_to_arabic,
    process_utterance
)

vowels = ['aa', 'AA', 'uu0', 'uu1', 'UU0', 'UU1', 'ii0', 'ii1',
          'II0', 'II1', 'a', 'A', 'u0', 'u1', 'U0', 'U1', 'i0', 'i1',
          'I0', 'I1']

vowel_map = {
    'aa': 'aa', 'AA': 'aa',
    'uu0': 'uu', 'uu1': 'uu', 'UU0': 'uu', 'UU1': 'uu',
    'ii0': 'ii', 'ii1': 'ii', 'II0': 'ii', 'II1': 'ii',
    'a': 'a', 'A': 'a',
    'u0': 'u', 'u1': 'u', 'U0': 'u', 'U1': 'u',
    'i0': 'i', 'i1': 'i', 'I0': 'i', 'I1': 'i'
}

phon_to_id_ = {phon: i for i, phon in enumerate(symbols)}
letter_to_id_ = {phon: i for i, phon in enumerate(symbols_full)}


def tokens_to_ids(phonemes, phon_to_id=None):
    if phon_to_id is None:
        phon_to_id = phon_to_id_
    
        token_ids = []
        for letter in phonemes:
            if letter in phon_to_id:
                token_ids.append(phon_to_id[letter])
                
    return token_ids


def ids_to_tokens(ids):
    return [symbols[id] for id in ids]


def arabic_to_phonemes(arabic):
    buckw = arabic_to_buckwalter(arabic)
    return process_utterance(buckw)


def buckwalter_to_phonemes(buckw):
    return process_utterance(buckw)


def phonemes_to_tokens(phonemes: str, append_space=True):
    phonemes = phonemes \
        .replace("sil", "") \
        .replace("+", "_+_") \
        .split()
    for i, phon in enumerate(phonemes):
        if len(phon) == 2 and phon not in vowels and phon[0] == phon[1]:
            phonemes[i] = phon[0]
            phonemes.insert(i+1, DOUBLING_TOKEN)
        if phonemes[i] in vowels:
            phonemes[i] = vowel_map[phonemes[i]]

    if append_space:
        phonemes.append(SEPARATOR_TOKEN)
   
    phonemes.append(EOS_TOKEN)

    return phonemes


def buckwalter_to_tokens(buckw, append_space=True):
    phonemes = buckwalter_to_phonemes(buckw)
    tokens = phonemes_to_tokens(phonemes, append_space=append_space)
    return tokens


def arabic_to_tokens(arabic, append_space=True):
    buckw = arabic_to_buckwalter(arabic)
    tokens = buckwalter_to_tokens(buckw, append_space=append_space)
    return tokens


def simplify_phonemes(phonemes):
    for k, v in vowel_map.items():
        phonemes = phonemes.replace(k, v)
    return phonemes


def tokenizer_phonetic(phrase, input_type='arabic', phon_to_id=None):
    
    if input_type == 'arabic':
        phonemes = arabic_to_phonemes(phrase)
    elif input_type == 'phonemes':
        phonemes = phrase
    elif input_type == 'buckwalter':
        phonemes = buckwalter_to_phonemes(phrase)
    
    tokens = phonemes_to_tokens(phonemes)
    
    token_ids = tokens_to_ids(tokens, phon_to_id=phon_to_id)
    
    return token_ids

def tokenizer_raw(phrase, letter_to_id=None):
    if letter_to_id is None:
        letter_to_id = letter_to_id_
    
    token_ids = []
    for letter in phrase:
        if letter in letter_to_id:
            token_ids.append(letter_to_id[letter])
    
    return token_ids
