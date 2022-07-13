from torch.utils.tensorboard import SummaryWriter
from utils.plotting import get_alignment_figure, get_specs_figure


class TBLogger(SummaryWriter):
    def __init__(self, log_dir):
        super(TBLogger, self).__init__(log_dir)

    def add_training_data(self, reduced_loss, grad_norm,
                          learning_rate, tb_step: int):

        self.add_scalar("loss/training_loss", reduced_loss, tb_step)
        self.add_scalar("training/grad_norm", grad_norm, tb_step)
        self.add_scalar("training/learning_rate", learning_rate, tb_step)

    def add_parameters(self, model, tb_step: int):

        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), tb_step)

    def add_sample(self, alignment, mel_pred,
                   mel_targ, mel_infer, len_targ,
                   tb_step: int):

        self.add_figure(
            "alignment",
            get_alignment_figure(alignment.detach().cpu().numpy().T),
            tb_step)

        self.add_figure(
            "spectrograms",
            get_specs_figure([
                mel_infer.detach().cpu().numpy(),
                mel_pred[:, :len_targ].detach().cpu().numpy(),
                mel_targ[:, :len_targ].detach().cpu().numpy(),
            ],
                ['Frames (inferred)', 'Frames (predicted)', 'Frames (target)']
            ), tb_step)
