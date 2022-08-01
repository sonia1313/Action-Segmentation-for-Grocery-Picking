import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pytorch_lightning as pl
from torchmetrics import Accuracy


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
    def __init__(self, n_features, n_classes, num_channels_per_level=[25] * 8, kernel_size=7, dropout=0.0):
        super().__init__()
        layers = []
        num_levels = len(num_channels_per_level)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = n_features if i == 0 else num_channels_per_level[i - 1]
            out_channels = num_channels_per_level[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        self.linear = nn.Linear(in_features=num_channels_per_level[-1], out_features=n_classes)

    def forward(self, x):
        out = self.network(x)
        # out shape : batch_size x hidden_size x seq_len e.g. 1 x 25 x 150

        x = out.view(-1, out.shape[1])  # x shape: seq_len x hidden

        logits = self.linear(x)  # logits shape: seq_len x n_classes

        return logits


class LitTemporalConvNet(pl.LightningModule):

    def __init__(self, n_features, n_hid, n_levels, kernel_size, dropout, lr, n_classes=6):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        self.num_channels = [n_hid] * n_levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr

        self.save_hyperparameters()

        self.tcn = TemporalConvNet(n_features=self.n_features, n_classes=self.n_classes, num_channels_per_level=self.num_channels,
                                   kernel_size=self.kernel_size,
                                   dropout=self.dropout)

        self.loss_module = nn.CrossEntropyLoss(ignore_index=-1)
        self.train_acc = Accuracy(ignore_index=-1, multiclass=True)
        self.val_acc = Accuracy(ignore_index=-1, multiclass=True)
        self.test_acc = Accuracy(ignore_index=-1, multiclass=True)

        # TODO: experiment tracking and other metrics

    def forward(self, x):
        logits = self.tcn(x)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch

        data = torch.permute(X, (0,2,1))

        logits = self(data)
        # logits shape?

        y = y[0][:].view(-1)
        loss = self.loss_module(logits, y)

        accuracy = self.train_acc(logits,y)

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        data = torch.permute(X, (0, 2, 1))

        logits = self(data)

        y = y[0][:].view(-1)

        loss = self.loss_module(logits, y)

        accuracy = self.val_acc(logits, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
