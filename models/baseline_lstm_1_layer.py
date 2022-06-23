from abc import ABC

import torch
import torch.nn as nn
import pytorch_lightning as pl


class ManyToManyLSTM(pl.LightningModule, ABC):
    def __init__(self, n_features=9, hidden_size=100, n_layers=1, n_classes=6):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes

        self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, batch_first=True)

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.n_classes)

        self.save_hyperparameters()

    def forward(self, x):
        batch_size = x.shape[0]
        #
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        #
        #h0, c0 = self._init_states(batch_size)

        output, (_, _) = self.lstm(x, (h0, c0))

        frames = output.view(-1, output.shape[2])  # flatten

        logits = self.linear(frames)

        return logits

    def _init_states(self, batch_size):
        if torch.cuda.is_available():
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True, device="cuda")
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True, device="cuda")

        else:
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True)

        return h0, c0
