import argparse
import os
import torch
import torchaudio
import text
import utils.make_html as html

from utils import progbar, read_lines_from_file

# default:
#python inference.py --list data/infer_text.txt --out_dir samples/results --model fastpitch --checkpoint pretrained/fastpitch_ar_adv.pth --batch_size 2 --denoise 0

# Examples:
#python inference.py --list data/infer_text.txt --out_dir samples/res_tc2_adv0 --model tacotron2 --checkpoint pretrained/tacotron2_ar_adv.pth --batch_size 2
#python inference.py --list data/infer_text.txt --out_dir samples/res_tc2_adv1 --model tacotron2 --checkpoint pretrained/tacotron2_ar_adv.pth --batch_size 2 --denoise 0.01
#python inference.py --list data/infer_text.txt --out_dir samples/res_fp_adv0 --model fastpitch --checkpoint pretrained/fastpitch_ar_adv.pth --batch_size 2
#python inference.py --list data/infer_text.txt --out_dir samples/res_fp_adv1 --model fastpitch --checkpoint pretrained/fastpitch_ar_adv.pth --batch_size 2 --denoise 0.01


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'fastpitch':
        from models.fastpitch import FastPitch2Wave
        model = FastPitch2Wave(args.checkpoint)
    elif args.model == 'tacotron2':
        from models.tacotron2 import Tacotron2Wave
        model = Tacotron2Wave(args.checkpoint)
    else:
        raise "model type not supported"

    model = model.to(device)
    model.eval()

    if not os.path.exists(f"{args.out_dir}/wavs"):
        os.makedirs(f"{args.out_dir}/wavs")

    static_lines = read_lines_from_file(args.list)
    static_batches = [static_lines[k:k+args.batch_size]
                      for k in range(0, len(static_lines), args.batch_size)]

    idx = 0
    with open(os.path.join(args.out_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html.make_html_start())

        for batch in progbar(static_batches):
            # infer batch
            wav_list = model.tts(batch,
                                 batch_size=args.batch_size,
                                 denoise=args.denoise,
                                 speed=args.speed)

            # save wavs and add entries to html file
            for (text_line, wav) in zip(batch, wav_list):
                torchaudio.save(f'{args.out_dir}/wavs/static{idx}.wav',
                                wav.unsqueeze(0),
                                22_050)

                text_buckw = text.arabic_to_buckwalter(text_line)
                text_arabic = text.buckwalter_to_arabic(text_buckw)
                t_phon = text.buckwalter_to_phonemes(text_buckw)
                t_phon = text.simplify_phonemes(
                    t_phon.replace(' ', '').replace('+', ' '))

                f.write(html.make_sample_entry2(
                    f'wavs/static{idx}.wav',
                    text_arabic,
                    f"{idx}) {t_phon}"))

                idx += 1

        f.write(html.make_volume_script(0.5))
        f.write(html.make_html_end())

    print(f"Saved files to: {args.out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--list', type=str, default='./data/infer_text.txt')
    parser.add_argument(
        '--model', type=str, default='fastpitch')
    parser.add_argument(
        '--checkpoint', type=str, default='pretrained/fastpitch_ar_adv.pth')
    parser.add_argument('--out_dir', type=str, default='samples/results')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--denoise', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', type=str,
                        default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    infer(args)


if __name__ == '__main__':
    main()
