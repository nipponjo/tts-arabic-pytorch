import random
import torch
import torch.nn.functional as F


def batch_to_device(batch, device):
    text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths = batch

    text_padded = text_padded.to(device, non_blocking=True)
    input_lengths = input_lengths.to(device, non_blocking=True)
    mel_padded = mel_padded.to(device, non_blocking=True)
    gate_padded = gate_padded.to(device, non_blocking=True)
    output_lengths = output_lengths.to(device, non_blocking=True)

    return (text_padded, input_lengths, mel_padded, gate_padded,
            output_lengths)


@torch.inference_mode()
def validate(model, test_loader, writer, device, n_iter):
    loss_sum = 0
    n_test_sum = 0

    model.eval()

    for batch in test_loader:
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch_to_device(batch, device)

        y_pred = model(text_padded, input_lengths,
                       mel_padded, output_lengths)
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
        text_padded[idx:idx+1], input_lengths[idx:idx+1])

    writer.add_sample(
        alignments[idx, :, :input_lengths[idx].item()],
        mel_out[idx], mel_padded[idx], mel_infer[0],
        output_lengths[idx], n_iter)

    writer.add_scalar('loss/val_loss', val_loss, n_iter)

    model.train()

    return val_loss


def save_states(fname, model, optimizer, n_iter, epoch, config):
    torch.save({'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': epoch,
                'iter': n_iter,
                },
               f'{config.checkpoint_dir}/{fname}')
