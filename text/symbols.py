
PADDING_TOKEN = '_pad_'
EOS_TOKEN = '_eos_'
DOUBLING_TOKEN = '_dbl_'
SEPARATOR_TOKEN = '_+_'

EOS_TOKENS = [SEPARATOR_TOKEN, EOS_TOKEN]

special_symbols = [
    # special tokens
    PADDING_TOKEN,  # padding
    EOS_TOKEN,  # eos-token
    '_sil_',  # silence
    DOUBLING_TOKEN,  # doubling
    SEPARATOR_TOKEN,  # word separator
]

punctuation = [
    # punctuation
    '.', ',', '?', '!', 
    '-', # hyphen-minus U+002D    
]

punctuation_arab = [
    'ـ', # (Arabic Tatweel) U+0640
    '_', # (low line) U+005F
    '؟', 
    '؛', 
    '،',
]

ligatures = [
'ۖ', # Arabic small high ligature Sad with Lam with Alef Maksura U+06D6

'ۗ', #  Arabic small high ligature Qaf with Lam with Alef Maksura U+06D7

 'ۘ', # Arabic small high Meem Initial Form U+06D8

 'ۚ',   # Arabic small high Jeem U+06DA
  
 'ۛ', #   Arabic small high three dots U+06DB

]

 
nawar_symbols = [
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




abc_arabic = list('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
abc_arab_hamza = ['ٱ','ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ى', 'ة']
# abc_arab_hamza += ['']

arab_vowels_short = [
    'َ', # a (fatha)
    'ِ', # i (kasra)
    'ُ', # u (damma)
]

arab_nunation = [
    'ً', # an
    'ٍ', # in
    'ٌ', # un
]

diacrit_an = 'ً' # an

diacrit_sukun = 'ْ' # sukun
diacrit_shadda = 'ّ' # shadda
diacrit_dagger_alif = 'ٰ' # dagger alif

diacrit_arab = [
    'َ', # a (fatha)
    'ِ', # i (kasra)
    'ُ', # u (damma)
    'ً', # an
    'ٍ', # in
    'ٌ', # un
    'ْ', # sukun
    'ّ', # shadda
    'ٰ', # dagger alif
]

abc_arab_extra = ['گ', 'چ', 'ڨ', 'ﻻ', 
                  'ک', # Arabic letter Keheh
                  'ی', # Arabic letter Farsi Yeh
                  'ھ', # Arabic letter Heh Doachashmee
                  ]
arab_special = ['ۖ', 'ۚ', 'ـ']

numerals = list('0123456789')
numerals_ar =  list('٠١٢٣٤٥٦٧٨٩')

symbols_nawar = special_symbols + nawar_symbols

symbols_full = special_symbols + punctuation + punctuation_arab + \
    nawar_symbols + abc_arabic + abc_arab_hamza + diacrit_arab + \
    abc_arab_extra + ligatures + numerals
    
symbols = symbols_nawar