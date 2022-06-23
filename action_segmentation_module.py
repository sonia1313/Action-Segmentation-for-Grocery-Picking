from abc import ABC
import time
import pytorch_lightning as pl
import torch.optim
import torch.nn as nn

from models.baseline_lstm_1_layer import ManyToManyLSTM


def _remove_padding_one_hot(predictions_padded, targets_padded):
    mask = (targets_padded >= 0).long()
    n = len([out for out in mask.squeeze() if out.all() >= 1])
    outputs = predictions_padded[:n, :]

    targets_padded = targets_padded.squeeze()
    targets = targets_padded[:n, :]
    _, targets = targets.max(dim=1)  # remove one hot encoding

    return outputs, targets


class ActionSegmentationModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ManyToManyLSTM()
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        padded_X, padded_y = batch
        padded_logits = self.model(padded_X)
        logits, y = _remove_padding_one_hot(padded_logits, padded_y)

        loss = self.loss_module(logits, y)

        # accuracy

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # padded_X, padded_y = batch
        # padded_preds = self.model(padded_X)
        # predictions, y = _remove_padding_one_hot(padded_preds, padded_y)
        # loss = self.loss_module(predictions, y)
        # self.log("val loss", loss)

        val_loss = self.training_step(batch,batch_idx)
        #time.sleep(1)

        self.log('val_loss', val_loss)
        return val_loss

    # def validation_epoch_end(self, val_step_results):
    #     #[val_result, val_result...]
    #
    #     avg_val_loss = torch.tensor([x['loss'] for x in val_step_results]).mean()
    #     pbar = {'avg_val_loss': avg_val_loss}
    #     #print(avg_val_loss)
    #     return {"val_loss" : avg_val_loss, 'progress_bar': pbar}

    def test_step(self, batch, batch_idx):
        # padded_X, padded_y = batch
        # padded_preds = self.model(padded_X)
        # predictions, y = _remove_padding_one_hot(padded_preds, padded_y)
        # loss = self.loss_module(predictions, y)

        test_loss = self.training_step(batch,batch_idx)
        self.log("test loss", test_loss)
        return test_loss
