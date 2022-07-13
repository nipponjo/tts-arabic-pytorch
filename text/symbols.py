
PADDING_TOKEN = '_pad_'
EOS_TOKEN = '_eos_'
DOUBLING_TOKEN = '_dbl_'
SEPARATOR_TOKEN = '_+_'

EOS_TOKENS = [SEPARATOR_TOKEN, EOS_TOKEN]

symbols = [
    # special tokens
    PADDING_TOKEN,  # padding
    EOS_TOKEN,  # eos-token
    '_sil_',  # silence
    DOUBLING_TOKEN,  # doubling
    SEPARATOR_TOKEN,  # word separator
    # consonants
    '<',  # hamza
    'b',  # baa'
    't',  # taa'
    '^',  # thaa'
    'j',  # jiim
    'H',  # Haa'
    'x',  # xaa'
    'd',  # daal
    '*',  # dhaal
    'r',  # raa'
    'z',  # zaay
    's',  # siin
    '$',  # shiin
    'S',  # Saad
    'D',  # Daad
    'T',  # Taa'
    'Z',  # Zhaa'
    'E',  # 3ayn
    'g',  # ghain
    'f',  # faa'
    'q',  # qaaf
    'k',  # kaaf
    'l',  # laam
    'm',  # miim
    'n',  # nuun
    'h',  # haa'
    'w',  # waaw
    'y',  # yaa'
    'v',  # /v/ for loanwords e.g. in u'fydyw': u'v i0 d y uu1',
    # vowels
    'a',  # short
    'u',
    'i',
    'aa',  # long
    'uu',
    'ii',
]
