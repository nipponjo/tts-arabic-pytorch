import sys
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class DictConfig(object):
    """Creates a Config object from a dict 
       such that object attributes correspond to dict keys.    
    """

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __str__(self):
        return '\n'.join(f"{key}: {val}" for key, val in self.__dict__.items())

    def __repr__(self):
        return self.__str__()


def get_custom_config(fname):
    with open(fname, 'r') as stream:
        config_dict = yaml.load(stream, Loader)
    config = DictConfig(config_dict)
    return config


def get_basic_config():
    return get_custom_config('configs/basic.yaml')


def get_config(fname):
    config = get_basic_config()
    custom_config = get_custom_config(fname)

    config.__dict__.update(custom_config.__dict__)
    return config


def read_lines_from_file(path, encoding='utf-8'):
    lines = []
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            lines.append(line.strip())
    return lines

def write_lines_to_file(path, lines, mode='w', encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for i, line in enumerate(lines):
            if i == len(lines)-1:
                f.write(line)
                break
            f.write(line + '\n')  
                      

def progbar(iterable, length=30, symbol='='):
    """Wrapper generator function for an iterable. 
       Prints a progressbar when yielding an item. \\
       Args:
          iterable: an object supporting iteration
          length: length of the progressbar
    """
    n = len(iterable)
    for i, item in enumerate(iterable):
        steps = length*(i+1) // n
        sys.stdout.write('\r')
        sys.stdout.write(f"[{symbol*steps:{length}}] {(100/n*(i+1)):.1f}%")
        if i == (n-1):
            sys.stdout.write('\n')
        sys.stdout.flush()
        yield item
