import torch
import torch.nn as nn
import wandb
from torch.nn.utils import weight_norm
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix

from utils.edit_distance import edit_score
from utils.metrics_utils import _get_average_metrics
from utils.overlap_f1_metric import f1_score
from utils.plot_confusion_matrix import _plot_cm
from utils.preprocessing import remove_padding


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # zero padding of length(kernel size−1) is added to ensure new layer is the same length as previous one.
        x = x[:, :, : -self.chomp_size].contiguous()
        return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, n_features, n_classes, num_channels_per_level, kernel_size, stride, dropout=0.0):
        super().__init__()
        layers = []
        num_levels = len(num_channels_per_level)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = n_features if i == 0 else num_channels_per_level[i - 1]
            out_channels = num_channels_per_level[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        self.linear = nn.Linear(in_features=num_channels_per_level[-1], out_features=n_classes)

    def forward(self, x):
        out = self.network(x)
        # out shape : batch_size x hidden_size x seq_len e.g. 1 x 25 x 150

        x = out.permute(0,2,1).squeeze(0) # x shape: seq_len x hidden

        logits = self.linear(x)  # logits shape: seq_len x n_classes

        return logits


class LitTemporalConvNet(pl.LightningModule):

    def __init__(self, n_features, n_hid, n_levels, kernel_size, dropout, lr,
                 exp_name, experiment_tracking=True,  n_classes=6, stride = 1):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        self.num_channels = [n_hid] * n_levels
        self.kernel_size = kernel_size
        #print(self.kernel_size)
        self.dropout = dropout
        self.lr = lr
        self.stride = stride

        self.save_hyperparameters()

        self.tcn = TemporalConvNet(n_features=self.n_features, n_classes=self.n_classes,
                                   num_channels_per_level=self.num_channels,
                                   kernel_size=self.kernel_size,
                                   dropout=self.dropout,
                                   stride = self.stride)

        self.loss_module = nn.CrossEntropyLoss(ignore_index=-1)
        self.train_acc = Accuracy(ignore_index=-1, multiclass=True)
        self.val_acc = Accuracy(ignore_index=-1, multiclass=True)
        self.test_acc = Accuracy(ignore_index=-1, multiclass=True)



        self.experiment_tracking = experiment_tracking
        self.test_counter = 0
        self.val_counter = 0
        self.confusion_matrix = ConfusionMatrix(num_classes=6)
        self.experiment_name = exp_name

    def forward(self, x):
        logits = self.tcn(x)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch

        data = torch.permute(X, (0, 2, 1))  # data is shape : B x C X L

        logits = self(data)

        y = y[0][:].view(-1)
        loss = self.loss_module(logits, y)

        accuracy = self.train_acc(logits, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding(logits, y)
        f1_scores = f1_score(preds, targets)
        edit = edit_score(preds, targets)

        cm = self.confusion_matrix(preds, targets)

        if self.experiment_tracking:
            wandb.log({"epoch": self.current_epoch, "train_loss": loss, "train_accuracy": accuracy,
                       "f1_overlap_10": f1_scores[0], "f1_overlap_25": f1_scores[1],
                       "f1_overlap_50": f1_scores[2], "edit_score": edit})
            if self.current_epoch % 5 == 0:
                cm_fig = _plot_cm(cm=cm,
                                  path=f"training_confusion_matrix/{self.experiment_name}-training-cm-{self.current_epoch}.png")
                wandb.log({"training_confusion_matrix": wandb.Image(cm_fig)})

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        data = torch.permute(X, (0, 2, 1))

        logits = self(data)

        y = y[0][:].view(-1)

        loss = self.loss_module(logits, y)

        accuracy = self.val_acc(logits, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding(logits, y)
        cm = self.confusion_matrix(preds, targets)

        f1_scores = f1_score(preds, targets)
        # print(f1_scores)
        edit = edit_score(preds, targets)

        fig = _plot_cm(cm, path=f"confusion_matrix_figs/{self.experiment_name}-validation-cm-{self.val_counter}.png")

        if self.experiment_tracking:
            wandb.log({"epoch": self.current_epoch, "val_loss": loss, "val_accuracy": accuracy,
                       "val_f1_overlap_10": float(f1_scores[0]), "val_f1_overlap_25": float(f1_scores[1]),
                       "val_f1_overlap_50": float(f1_scores[2]), "val_edit_score": edit})
            if self.current_epoch % 5 == 0:
                wandb.log({"validation_confusion_matrix": wandb.Image(fig)})
        return accuracy, edit, f1_scores

    def validation_epoch_end(self, outputs):

        f1_10_mean, f1_25_mean, f1_50_mean, edit_mean, accuracy_mean = _get_average_metrics(outputs)

        if self.experiment_tracking:
            wandb.log({"average_val_f1_10": f1_10_mean, "average_val_f1_25": f1_25_mean,
                       "average_val_f1_50": f1_50_mean, "average_val_edit": edit_mean,
                       "average_val_accuracy": accuracy_mean})

    def test_step(self, batch, batch_idx):
        X, y = batch

        data = torch.permute(X, (0, 2, 1))
        logits = self(data)

        y = y[0][:].view(-1)

        loss = self.loss_module(logits, y)

        accuracy = self.test_acc(logits, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding(logits, y)
        # print(f"preds shape: {preds.shape}")
        # print(f"target shape: {targets.shape}")
        cm = self.confusion_matrix(preds, targets)

        f1_scores = f1_score(preds, targets)
        # print(f1_scores)
        edit = edit_score(preds, targets)

        fig = _plot_cm(cm, path=f"confusion_matrix_figs/{self.experiment_name}-test-{self.test_counter}.png")

        if self.experiment_tracking:
            wandb.log({"test_loss": loss, "test_accuracy": accuracy, "test_confusion_matrix": wandb.Image(fig)})

        return accuracy, edit, f1_scores

    def test_epoch_end(self, outputs):

        f1_10_mean, f1_25_mean, f1_50_mean, edit_mean, accuracy_mean = _get_average_metrics(outputs)

        print(f"average test f1 overlap @ 10%: {f1_10_mean}")
        print(f"average test f1 overlap @ 25%: {f1_25_mean}")
        print(f"average test f1 overlap @ 50%: {f1_50_mean}")
        print(f"average test edit: {edit_mean}")
        print(f"average test accuracy : {accuracy_mean}")

        if self.experiment_tracking:
            wandb.log({"average_test_f1_10": f1_10_mean, "average_test_f1_25": f1_25_mean,
                       "average_test_f1_50": f1_50_mean, "average_test_edit": edit_mean,
                       "average_test_accuracy": accuracy_mean})