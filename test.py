import argparse
import os
import torch
import torchaudio

import text
import utils.make_html as html
from utils.plotting import get_spectrogram_figure
from vocoder import load_hifigan
from vocoder.hifigan.denoiser import Denoiser
from utils import get_basic_config

#default:
#python test.py --model fastpitch --checkpoint pretrained/fastpitch_ar_adv.pth --out_dir samples/test

# Examples:
#python test.py --model fastpitch --checkpoint pretrained/fastpitch_ar_adv.pth --out_dir samples/test_fp_adv
#python test.py --model fastpitch --checkpoint pretrained/fastpitch_ar_adv.pth --denoise 0.01 --out_dir samples/test_fp_adv_d
#python test.py --model fastpitch --checkpoint pretrained/fastpitch_ar_mse.pth --out_dir samples/test_fp_mse

#python test.py --model tacotron2 --checkpoint pretrained/tacotron2_ar_adv.pth --out_dir samples/test_tc2_adv
#python test.py --model tacotron2 --checkpoint pretrained/tacotron2_ar_adv.pth --denoise 0.01 --out_dir samples/test_tc2_adv_d
#python test.py --model tacotron2 --checkpoint pretrained/tacotron2_ar_mse.pth --out_dir samples/test_tc2_mse


def test(args, text_arabic):

    config = get_basic_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = args.out_dir
    sample_rate = 22_050

    # Load model
    if args.model == 'fastpitch':
        from models.fastpitch import FastPitch
        model = FastPitch(args.checkpoint)
    elif args.model == 'tacotron2':
        from models.tacotron2 import Tacotron2
        model = Tacotron2(args.checkpoint)
    else:
        raise "model type not supported"

    print(f'Loaded {args.model} from: {args.checkpoint}')
    model.eval()

    # Load vocoder model
    vocoder = load_hifigan(
        state_dict_path=config.vocoder_state_path,
        config_file=config.vocoder_config_path)
    print(f'Loaded vocoder from: {config.vocoder_state_path}')

    model, vocoder = model.to(device), vocoder.to(device)
    denoiser = Denoiser(vocoder)

    # Infer spectrogram and wave
    with torch.inference_mode():
        mel_spec = model.ttmel(text_arabic)
        wave = vocoder(mel_spec[None])
        if args.denoise > 0:
            wave = denoiser(wave, args.denoise)            

    # Save wave and images
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created folder: {out_dir}")

    torchaudio.save(f'{out_dir}/wave.wav', wave[0].cpu(), sample_rate)

    get_spectrogram_figure(mel_spec.cpu()).savefig(
        f'{out_dir}/mel_spec.png')

    t_phon = text.arabic_to_phonemes(text_arabic)
    t_phon = text.simplify_phonemes(t_phon.replace(' ', '').replace('+', ' '))

    with open(f'{out_dir}/index.html', 'w', encoding='utf-8') as f:
        f.write(html.make_html_start())
        f.write(html.make_h_tag("Test sample", n=1))
        f.write(html.make_sample_entry2(f"./wave.wav", text_arabic, t_phon))
        f.write(html.make_h_tag("Spectrogram"))
        f.write(html.make_img_tag('./mel_spec.png'))
        f.write(html.make_volume_script(0.42))
        f.write(html.make_html_end())

    print(f"Saved test sample to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='fastpitch')
    parser.add_argument(
        '--checkpoint', default='pretrained/fastpitch_ar_adv.pth')  
    parser.add_argument('--denoise', type=float, default=0)  
    parser.add_argument('--out_dir', default='samples/test')
    args = parser.parse_args()

    text_arabic = "أَلسَّلامُ عَلَيكُم يا صَديقي"

    test(args, text_arabic)


if __name__ == '__main__':
    main()
