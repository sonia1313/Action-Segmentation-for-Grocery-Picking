from abc import ABC
import time
import pytorch_lightning as pl
import torch.optim
import torch.nn as nn
from torchmetrics import Accuracy

from models.baseline_lstm_1_layer import ManyToManyLSTM


def _remove_padding_one_hot(predictions_padded, targets_padded):
    mask = (targets_padded >= 0).long()
    n = len([out for out in mask.squeeze() if out.all() >= 1])
    outputs = predictions_padded[:n, :]

    targets_padded = targets_padded.squeeze()
    targets = targets_padded[:n]
    #_, targets = targets.max(dim=1)  # remove one hot encoding

    return outputs, targets


class ActionSegmentationModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ManyToManyLSTM()
        self.loss_module = nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        padded_X, padded_y = batch
        padded_logits = self.model(padded_X)
        logits, y = _remove_padding_one_hot(padded_logits, padded_y)
        #train_acc = (logits.argmax(dim=-1) == y).float().mean()

        #self.log("train_acc", train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.train_acc(logits, y)
        loss = self.loss_module(logits, y)

        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        padded_X, padded_y = batch
        padded_logits = self.model(padded_X)
        logits, y = _remove_padding_one_hot(padded_logits, padded_y)

        #val_acc = (logits.argmax(dim=-1) == y).float().mean()
        self.val_acc(logits,y)
        val_loss = self.loss_module(logits, y)

        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', val_loss, on_step=False,  on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        padded_X, padded_y = batch
        padded_logits = self.model(padded_X)
        logits, y = _remove_padding_one_hot(padded_logits, padded_y)

        #test_acc = (logits.argmax(dim=-1) == y).float().mean()

        test_loss = self.loss_module(logits, y)
        # test_loss = self.training_step(batch,batch_idx)

        self.test_acc(logits,y)

        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return test_loss
