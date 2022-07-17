from abc import ABC
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy


class ManyToManyLSTM(nn.Module):
    def __init__(self, n_features=3, hidden_size=100, n_layers=1, n_classes=6):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes

        self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, batch_first=True)

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.n_classes)

        # self.save_hyperparameters()

    def forward(self, x):
        batch_size = x.shape[0]
        #
        # h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        # c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        # #
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

    def __init__(self, n_features, hidden_size, n_layers):
        super().__init__()

        self.lstm = ManyToManyLSTM(n_features, hidden_size, n_layers)
        self.loss_module = nn.CrossEntropyLoss(ignore_index=-1)
        self.train_acc = Accuracy(ignore_index=-1)
        self.val_acc = Accuracy(ignore_index=-1)
        self.test_acc = Accuracy(ignore_index=-1)

    def forward(self, X):
        logits = self.lstm(X)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())

        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch

        logits = self(X)
        logits = logits.squeeze(0)
        y = y.squeeze(0)
        loss = self.loss_module(logits, y)

        self.train_acc(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        X, y = batch

        logits = self(X)
        logits = logits.squeeze(0)
        y = y.squeeze(0)
        loss = self.loss_module(logits, y)

        self.val_acc(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y = batch

        logits = self(X)
        logits = logits.squeeze(0)
        y = y.squeeze(0)
        loss = self.loss_module(logits, y)

        self.test_acc(logits, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
