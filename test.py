import argparse
import os
import torch
import torchaudio

import text
import utils.make_html as html
from utils.plotting import get_alignment_figure, get_spectrogram_figure
from vocoder import load_hifigan
from utils import get_basic_config
from model.tacotron2_ms import Tacotron2MS

def test(args, text_arabic):

    config = get_basic_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = args.out_dir
    sample_rate = 22_050

    # Load tacotron2 model
    model = Tacotron2MS(n_symbol=40)
    model_state_dict = torch.load(args.checkpoint)['model']
    model.load_state_dict(model_state_dict)
    print(f'Loaded tacotron2 from: {args.checkpoint}')
    model.eval()

    # Load vocoder model
    vocoder = load_hifigan(
        state_dict_path=config.vocoder_state_path,
        config_file=config.vocoder_config_path)
    print(f'Loaded vocoder from: {config.vocoder_state_path}')

    model, vocoder = model.to(device), vocoder.to(device)

    # Process utterance
    tokens = text.arabic_to_tokens(text_arabic)
    token_ids = text.tokens_to_ids(tokens)

    ids_batch = torch.LongTensor(token_ids).unsqueeze(0).to(device)

    # Infer spectrogram and wave
    with torch.inference_mode():
        mel_spec, _, alignments = model.infer(ids_batch)
        wave = vocoder(mel_spec)

    # Save wave and images
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created folder: {out_dir}")

    torchaudio.save(f'{out_dir}/wave.wav', wave[0].cpu(), sample_rate)

    get_spectrogram_figure(mel_spec[0].cpu()).savefig(
        f'{out_dir}/mel_spec.png')
    get_alignment_figure(alignments[0].cpu().t()).savefig(
        f'{out_dir}/alignment.png')

    t_phon = text.arabic_to_phonemes(text_arabic)
    t_phon = text.simplify_phonemes(t_phon.replace(' ', '').replace('+', ' '))

    with open(f'{out_dir}/index.html', 'w', encoding='utf-8') as f:
        f.write(html.make_html_start())
        f.write(html.make_h_tag("Test sample", n=1))
        f.write(html.make_sample_entry2(f"./wave.wav", text_arabic, t_phon))
        f.write(html.make_h_tag("Spectrogram"))
        f.write(html.make_img_tag('./mel_spec.png'))
        f.write(html.make_h_tag("Alignment"))
        f.write(html.make_img_tag('./alignment.png'))
        f.write(html.make_html_end())

    print(f"Saved test sample to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', default='pretrained/tacotron2_ar_adv.pth')
    parser.add_argument('--out_dir', default='samples/test')
    args = parser.parse_args()

    text_arabic = "أَلسَّلامُ عَلَيكُم يا صَديقي"

    test(args, text_arabic)


if __name__ == '__main__':
    main()
