import torch
from .symbols import input_vocab_to_int, output_int_to_vocab


def combine_text_with_harakat(input_sent: str, output_sent: str):
    #fix combine differences
    input_length  = len(input_sent)
    output_length = len(output_sent) # harakat_stack.size()
    for _ in range(0,(input_length-output_length)):
        output_sent.append("")

    #combine with text
    text = ""
    for character, haraka in zip(input_sent, output_sent):
        if haraka == '<UNK>' or haraka == 'ـ':
            haraka = ''
        text += character + "" + haraka

    return text

def encode(input_text: str, max_sentence: int=315):
    input_letters_ids  = [input_vocab_to_int.get(ch, input_vocab_to_int['<UNK>']) for ch in input_text]
    if max_sentence is not None:
        input_ids_pad = input_letters_ids + (max_sentence - len(input_letters_ids))*[0,]
    else:
        input_ids_pad = input_letters_ids
    return input_ids_pad, input_letters_ids

def decode(probs, text_input: str, input_letters_ids):
    diacrits = [output_int_to_vocab[i] for i in torch.argmax(probs[0], dim=1).tolist()[:len(input_letters_ids)]]
    return combine_text_with_harakat(text_input, diacrits)
