import argparse
import os
import torch
import torchaudio
import text
import utils.make_html as html
from model.networks import Tacotron2Wave
from utils import progbar, read_lines_from_file


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Tacotron2Wave(args.checkpoint)
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

        f.write(html.make_volume_script())
        f.write(html.make_html_end())

    print(f"Saved files to: {args.out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--list', type=str, default='./data/infer_text.txt')
    parser.add_argument(
        '--checkpoint', type=str, default='pretrained/tacotron2_ar_adv.pth')
    parser.add_argument('--out_dir', type=str, default='samples/results')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str,
                        default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    infer(args)


if __name__ == '__main__':
    main()
