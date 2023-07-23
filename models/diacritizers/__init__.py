from .shakkelha.network import Shakkelha
from .shakkala.network import Shakkala

def load_vowelizer(name: str, config):
    if name == 'shakkala':        
        shakkala = Shakkala(sd_path=config.shakkala_path)
        return shakkala
    elif name == 'shakkelha':
        shakkelha = Shakkelha(sd_path=config.shakkelha_path)
        return shakkelha
    else:
        print('...')