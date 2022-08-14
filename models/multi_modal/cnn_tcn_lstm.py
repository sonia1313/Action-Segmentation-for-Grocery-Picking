"""
Author - Sonia Mathews
cnn_tcn_lstm.py
"""
import torch
import torch.nn as nn
import wandb
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix

from models.cnn_tcn import CNN_TCN
from models.lstm import ManyToManyLSTM
from utils.edit_distance import edit_score
from utils.image_preprocessing import remove_padding_img
from utils.metrics_utils import _get_average_metrics
from utils.overlap_f1_metric import f1_score
from utils.plot_confusion_matrix import _plot_cm


class LitMM_CNN_TCN_LSTM(pl.LightningModule):
    def __init__(self, lr, lstm_n_features, lstm_nhid, lstm_nlayers, lstm_dropout, cnn_kernel_size, tcn_nhid,
                 tcn_levels, tcn_kernel_size, tcn_dropout, exp_name, experiment_tracking = True, n_classes=6):
        super().__init__()
        # self.tcn_num_channels = [tcn_hid] * tcn_levels
        self.cnn_tcn = CNN_TCN(cnn_kernel_size=cnn_kernel_size,
                               tcn_hid=tcn_nhid,
                               tcn_levels=tcn_levels,
                               tcn_kernel_size=tcn_kernel_size,
                               tcn_dropout=tcn_dropout
                               )
       # print(lstm_dropout)
        self.lstm = ManyToManyLSTM(n_features=lstm_n_features,
                                   hidden_size=lstm_nhid,
                                   n_layers=lstm_nlayers,
                                   dropout=lstm_dropout)

        combined_in_features = lstm_nhid + tcn_nhid
        self.action_fc1 = nn.Linear(in_features=combined_in_features, out_features=n_classes)

        self.save_hyperparameters()
        self.lr = lr
        self.loss_module = nn.CrossEntropyLoss(ignore_index=-1)
        self.train_acc = Accuracy(ignore_index=-1, multiclass=True)
        self.val_acc = Accuracy(ignore_index=-1, multiclass=True)
        self.test_acc = Accuracy(ignore_index=-1, multiclass=True)

        self.test_counter = 0
        self.val_counter = 0
        self.confusion_matrix = ConfusionMatrix(num_classes=6)

        self.experiment_name = exp_name
        self.experiment_tracking = experiment_tracking

    def forward(self, tactile_data, image_data):
        _, lstm_hidden = self.lstm(tactile_data)
        _, cnn_tcn_hidden = self.cnn_tcn(image_data)
        # print(lstm_hidden.shape)
        # print(lstm_hidden.shape)

        combined_hidden = torch.concat((cnn_tcn_hidden, lstm_hidden), dim=1)  # image and tactile data combined
        # print(combined_hidden.shape)
        # last_combined_hidden = combined_hidden[-1:] #last timestep
        # print(last_combined_hidden.shape)

        action_logits = self.action_fc1(combined_hidden)
        return action_logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, batch, batch_idx):
        image_x, tactile_x, y, fruit = batch

        cnn_in = image_x.flatten(start_dim=0, end_dim=1)
        lstm_in = tactile_x

        logits = self(tactile_data=lstm_in, image_data=cnn_in)

        y = y.squeeze(0)
        #print(action_logits.shape)
        #print(y.shape)
        loss = self.loss_module(logits, y)


        accuracy = self.train_acc(logits, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding_img(logits, y)
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
        image_x, tactile_x, y, fruit = batch

        cnn_in = image_x.flatten(start_dim=0, end_dim=1)
        lstm_in = tactile_x
        logits = self(tactile_data=lstm_in, image_data=cnn_in)

        # y = y[0][:].view(-1)
        y = y.squeeze(0)
        loss = self.loss_module(logits, y)

        accuracy = self.val_acc(logits, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding_img(logits, y)
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
        image_x, tactile_x, y, fruit = batch

        cnn_in = image_x.flatten(start_dim=0, end_dim=1)
        lstm_in = tactile_x
        logits = self(tactile_data=lstm_in, image_data=cnn_in)

        # y = y[0][:].view(-1)
        y = y.squeeze(0)
        loss = self.loss_module(logits, y)

        accuracy = self.test_acc(logits, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding_img(logits, y)
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

# def main():
#
#     # ----testing-cnn-lstm
#
#     model = LitMM_CNN_TCN_LSTM(lr=0.0,lstm_n_features=3,lstm_nhid=75,lstm_nlayers=1,
#                                lstm_dropout=0.0,
#                                cnn_kernel_size=3,tcn_nhid=75,tcn_levels=1,
#                                tcn_kernel_size=4,tcn_dropout=0.0)
#     batch_size = 1
#     time_steps = 100
#     n_channels = 3
#     size = 32
#     dummy_image_x = torch.rand(batch_size, time_steps, n_channels, size, size)
#     cnn_in = dummy_image_x.flatten(start_dim=0, end_dim=1)
#
#     dummy_tactile_x = torch.rand(batch_size,time_steps,n_channels)
#     lstm_in = dummy_tactile_x
#     action_logits, fruit_logits = model(tactile_data=lstm_in, image_data=cnn_in)
#     print(action_logits.shape)
#
#
#
#
# if __name__ == '__main__':
#     main()
