import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.tacotron2.tacotron2_ms import Tacotron2MS

from utils import get_config
from utils.data import ArabDataset, text_mel_collate_fn
from utils.logging import TBLogger
from utils.training import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default="configs/nawar.yaml", help="Path to yaml config file")


@torch.inference_mode()
def validate(model, test_loader, writer, device, n_iter):
    loss_sum = 0
    n_test_sum = 0

    model.eval()

    for batch in test_loader:
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch_to_device(batch, device)

        y_pred = model(text_padded, input_lengths,
                       mel_padded, output_lengths,
                       torch.zeros_like(output_lengths))                       
        mel_out, mel_out_postnet, gate_pred, alignments = y_pred

        mel_loss = F.mse_loss(mel_out, mel_padded) + \
            F.mse_loss(mel_out_postnet, mel_padded)
        gate_loss = F.binary_cross_entropy_with_logits(gate_pred, gate_padded)
        loss = mel_loss + gate_loss

        loss_sum += mel_padded.size(0)*loss.item()
        n_test_sum += mel_padded.size(0)

    val_loss = loss_sum / n_test_sum

    idx = random.randint(0, mel_padded.size(0) - 1)
    mel_infer, *_ = model.infer(
        text_padded[idx:idx+1], input_lengths[idx:idx+1]*0, input_lengths[idx:idx+1])

    writer.add_sample(
        alignments[idx, :, :input_lengths[idx].item()],
        mel_out[idx], mel_padded[idx], mel_infer[0],
        output_lengths[idx], n_iter)

    writer.add_scalar('loss/val_loss', val_loss, n_iter)

    model.train()

    return val_loss


def training_loop(model,
                  optimizer,
                  train_loader,
                  test_loader,
                  writer,
                  device,
                  config,
                  n_epoch,
                  n_iter):

    model.train()

    for epoch in range(n_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        for batch in train_loader:

            text_padded, input_lengths, mel_padded, gate_padded, \
                output_lengths = batch_to_device(batch, device)

            y_pred = model(text_padded, input_lengths,
                           mel_padded, output_lengths,
                           torch.zeros_like(output_lengths))
            mel_out, mel_out_postnet, gate_out, _ = y_pred

            optimizer.zero_grad()

            # LOSS
            mel_loss = F.mse_loss(mel_out, mel_padded) + \
                F.mse_loss(mel_out_postnet, mel_padded)
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_out, gate_padded)
            loss = mel_loss + gate_loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_thresh)
            optimizer.step()

            # LOGGING
            print(f"loss: {loss.item()}, grad_norm: {grad_norm.item()}")

            writer.add_training_data(loss.item(), grad_norm.item(),
                                     config.learning_rate, n_iter)

            if n_iter % config.n_save_states_iter == 0:
                save_states(f'states.pth', model, optimizer,
                            n_iter, epoch, config)

            if n_iter % config.n_save_backup_iter == 0 and n_iter > 0:
                save_states(f'states_{n_iter}.pth', model,
                            optimizer, n_iter, epoch, config)

            n_iter += 1

        # VALIDATE
        val_loss = validate(model, test_loader, writer, device, n_iter)
        print(f"Validation loss: {val_loss}")

        save_states(f'states_{n_iter}.pth', model,
                    optimizer, n_iter, epoch, config)


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set random seed
    if config.random_seed != False:
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        import numpy as np
        np.random.seed(config.random_seed)

    # make checkpoint folder if nonexistent
    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(os.path.abspath(config.checkpoint_dir))
        print(f"Created checkpoint_dir folder: {config.checkpoint_dir}")

    # datasets
    if config.cache_dataset:
        print('Caching datasets ...')
    train_dataset = ArabDataset(config.train_labels, config.train_wavs_path,
                                cache=config.cache_dataset)
    test_dataset = ArabDataset(config.test_labels, config.test_wavs_path,
                               cache=config.cache_dataset)

    # optional: balanced sampling
    sampler, shuffle, drop_last = None, True, True
    if config.balanced_sampling:
        weights = torch.load(config.sampler_weights_file)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=False)
        shuffle, drop_last = False, False

    # dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              collate_fn=text_mel_collate_fn,
                              shuffle=shuffle, drop_last=drop_last,
                              sampler=sampler)

    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size, drop_last=False,
                             shuffle=False, collate_fn=text_mel_collate_fn)

    # construct model
    model = Tacotron2MS(n_symbol=40)
    model = model.to(device)
    model.decoder.decoder_max_step = config.decoder_max_step

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    # resume from existing checkpoint
    n_epoch, n_iter = 0, 0
    if config.restore_model != '':
        state_dicts = torch.load(config.restore_model)
        model.load_state_dict(state_dicts['model'])
        if 'optim' in state_dicts:
            optimizer.load_state_dict(state_dicts['optim'])
        if 'epoch' in state_dicts:
            n_epoch = state_dicts['epoch']
        if 'iter' in state_dicts:
            n_iter = state_dicts['iter']

    # tensorboard writer
    writer = TBLogger(config.log_dir)

    # start training
    training_loop(model,
                  optimizer,
                  train_loader,
                  test_loader,
                  writer,
                  device,
                  config,
                  n_epoch,
                  n_iter)


if __name__ == '__main__':
    main()
