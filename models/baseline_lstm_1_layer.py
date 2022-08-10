from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import wandb
from torchmetrics import Accuracy, ConfusionMatrix

from utils.edit_distance import edit_score
from utils.metrics_utils import _get_average_metrics
from utils.overlap_f1_metric import f1_score
from utils.plot_confusion_matrix import _plot_cm
from utils.tactile_preprocessing import remove_padding

import wandb

class ManyToManyLSTM(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers, dropout, n_classes=6):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes

        self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, batch_first=True, dropout=dropout)

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.n_classes)

        # self.save_hyperparameters()

    def forward(self, x):
        batch_size = x.shape[0]

        h0, c0 = self._init_states(batch_size)

        output, (h_n, c_n) = self.lstm(x, (h0, c0))
        # shape of output: (1,142,100) - corresponds to a hidden state at each time step
        # shape of h_n: (1,1,100) - corresponds to the hidden_state from the last time time step only

        frames = output.view(-1, output.shape[2])  # flatten
        # shape of frames: (142,100)
        logits = self.linear(frames)
        # shape of logits: (142,6)

        return logits

    def _init_states(self, batch_size):
        if torch.cuda.is_available():
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True, device="cuda")
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True, device="cuda")

        else:
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True)

        return h0, c0


class LitManyToManyLSTM(pl.LightningModule):

    def __init__(self, n_features, hidden_size, n_layers, dropout, exp_name, lr, experiment_tracking=True):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = ManyToManyLSTM(n_features=n_features, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
        self.lr = lr

        self.loss_module = nn.CrossEntropyLoss(ignore_index=-1)
        self.train_acc = Accuracy(ignore_index=-1, multiclass=True)
        self.val_acc = Accuracy(ignore_index=-1, multiclass=True)
        self.test_acc = Accuracy(ignore_index=-1, multiclass=True)

        self.experiment_tracking = experiment_tracking
        self.test_counter = 0
        self.val_counter = 0
        self.confusion_matrix = ConfusionMatrix(num_classes=6)
        self.experiment_name = exp_name

    def forward(self, X):
        logits = self.lstm(X)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch

        logits, loss = self._get_preds_and_loss(X, y)
        #y = y[0][:].view(-1)  # shape = [max_seq_len]
        y = y.squeeze(0)
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

        self.val_counter += 1
        # print(self.val_counter)
        logits, val_loss = self._get_preds_and_loss(X, y)
        #y = y[0][:].view(-1)  # shape = [max_seq_len]
        y = y.squeeze(0)
        accuracy = self.val_acc(logits, y)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding(logits, y)
        cm = self.confusion_matrix(preds, targets)

        f1_scores = f1_score(preds, targets)
        # print(f1_scores)
        edit = edit_score(preds, targets)

        fig = _plot_cm(cm, path=f"confusion_matrix_figs/{self.experiment_name}-validation-cm-{self.val_counter}.png")

        if self.experiment_tracking:
            wandb.log({"epoch": self.current_epoch, "val_loss": val_loss, "val_accuracy": accuracy,
                       "val_f1_overlap_10": float(f1_scores[0]), "val_f1_overlap_25": float(f1_scores[1]),
                       "val_f1_overlap_50": float(f1_scores[2]), "val_edit_score": edit})
            if self.current_epoch % 5 == 0:
                wandb.log({"validation_confusion_matrix": wandb.Image(fig)})
        # return {"val_accuracy": accuracy, "val_edit": edit_score, "val_f1_scores": f1_scores}
        return accuracy, edit, f1_scores

    def validation_epoch_end(self, outputs):

        f1_10_mean, f1_25_mean, f1_50_mean, edit_mean, accuracy_mean = _get_average_metrics(outputs)

        if self.experiment_tracking:
            wandb.log({"average_val_f1_10": f1_10_mean, "average_val_f1_25": f1_25_mean,
                       "average_val_f1_50": f1_50_mean, "average_val_edit": edit_mean,
                       "average_val_accuracy": accuracy_mean})

    def test_step(self, batch, batch_idx):

        X, y = batch
        # print(X[0][0])
        self.test_counter += 1
        # print(f"test:{self.test_counter}")

        logits, test_loss = self._get_preds_and_loss(X, y)
        #y = y[0][:].view(-1)  # shape = [max_seq_len]
        y = y.squeeze(0)
        accuracy = self.test_acc(logits, y) 

        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding(logits, y)



        cm = self.confusion_matrix(preds, targets)

        f1_scores = f1_score(preds, targets)
        # print(f1_scores)
        edit = edit_score(preds, targets)

        fig = _plot_cm(cm, path=f"confusion_matrix_figs/{self.experiment_name}-test-{self.test_counter}.png")

        if self.experiment_tracking:
            wandb.log({"test_loss": test_loss, "test_accuracy": accuracy, "test_confusion_matrix": wandb.Image(fig)})

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

    def _get_preds_and_loss(self, X, y):

        logits = self(X)

        # print(logits.shape)
        # print(y.shape)

        

        # logits: batch_size,max_seqlen-1,n_classes e.g.[1,194,6]
        # y: batch_size,max_seqlen-1 e.g. [1,194]
        

        y = y[0][:].view(-1)
        # logits: max_seqlen-1,n_classes e.g.[194,6]
        # y: max_seqlen-1 e.g. [194]

        logits = logits.type_as(X)

        loss = self.loss_module(logits, y)

        return logits, loss


