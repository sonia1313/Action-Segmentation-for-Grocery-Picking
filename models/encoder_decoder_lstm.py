import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import random


class EncoderLSTM(nn.Module):
    """ Encodes tactile time series data """

    def __init__(self, n_features=3, hidden_size=100, n_layers=1, n_classes=6):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes

        self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, batch_first=True)

    def forward(self, x):
        batch_size = x.shape[0]

        output, (h_n, c_n) = self.lstm(x)
        # print(f"encoder hidden shape {h_n.shape}")
        # print(f"encoder cell shape {c_n.shape}")
        #
        # print(f"output shape {output.shape}")
        return h_n, c_n


class DecoderLSTM(torch.nn.Module):
    def __init__(self, hidden_size=100, n_classes=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        self.lstm = nn.LSTM(1, self.hidden_size)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.n_classes)

    def forward(self, x, hidden, cell):
        """"
        hidden - the final hidden state from the encoder model is the context vector of the source sequence.
        x - is the target ouput

        """
        # x = x.unsqueeze(0)
        # print("adding extra dimension in decoder input")
        # x = torch.LongTensor(x.view(1,1,1))
        #
        # print(f"decoder input shape: {x.shape}")
        # lstm input size: (seq_len, batch,n_features) = (1,1,1)
        # print(f"decoder input: {x}")
        # print(f"decoder hidden input shape: {hidden.shape}")
        # print(f"decoder cell input shape : {cell.shape}")
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        # output shape: (1,1,100)

        # print(f"decoder output shape: {output.shape}")
        # print(f"decoder hidden shape: {hidden.shape}")
        # print(f"decoder cell shape: {hidden.shape}")

        flatten_output = output.view(-1, output.shape[2])

        # print(f"flatten output shape:  {flatten_output.shape} ")
        # flatten_output shape: (1, 100)
        logits = self.linear(flatten_output)
        # shape of logits: (1,6)

        # print(f"logits shape:{logits.shape}")

        return logits, hidden, cell


# encoder gives a context vector
# decoder uses context vector to predict h0, h1, h2...hn
# decoder can also use teacher forcing [h0,y0],[h1,y1],[h2,y2]
class EncoderDecoderLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderLSTM()
        self.decoder = DecoderLSTM()

        assert self.encoder.hidden_size == self.decoder.hidden_size

    def forward(self, x, y, teacher_forcing_ratio, n_classes=6):
        batch_size = y.shape[0]
        optoforce_seq_len = y.shape[1]

        outputs = torch.zeros(batch_size, optoforce_seq_len, n_classes)

        hidden, cell = self.encoder(x)
        # print("context vector shape")
        # print(hidden.shape)
        # decoder_input = torch.zeros((1,1,1)) #not sure
        # print("beginning decoder input")
        # print(decoder_input.shape)
        decoder_input = y[0][0]
        decoder_input = decoder_input.type(torch.float32)
        decoder_input = decoder_input.view(1, 1, 1)
        for t in range(1, optoforce_seq_len):
            # print(t)
            # print(f"decoder input shape at time {t} = {decoder_input.shape}")
            # print(f"h_x.shape {hidden.shape}")
            # print(f"cell.shape {cell.shape}")
            # print(f"current t {t}")
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            # output shape (1,6)

            # print(f"output predicted at time {t} = {output}")
            outputs[0][t - 1] = output

            teacher_force = random.random() < teacher_forcing_ratio
            # print(f"teacher forcing present at time {t} = {teacher_force}")
            top_pred = output.argmax(-1)
            # print(f"top predicition at time {t} = {top_pred}")
            decoder_input = y[0][t] if teacher_force else top_pred

            # print(f"getting decoder input from time {t}")
            decoder_input = decoder_input.type(torch.float32)
            decoder_input = decoder_input.view(1, 1, 1)
            # print(f"next input at time {t} = {decoder_input.shape}, {decoder_input}")
            #

        return outputs


class LitEncoderDecoderLSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder_decoder_model = EncoderDecoderLSTM()
        self.loss_module = nn.CrossEntropyLoss(ignore_index=-1)
        self.train_acc = Accuracy(ignore_index=-1)
        self.val_acc = Accuracy(ignore_index=-1)
        self.test_acc = Accuracy(ignore_index=-1)

    def forward(self, X, y, teacher_forcing):
        logits = self.encoder_decoder_model(X, y, teacher_forcing)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch

        logits, loss = self._get_preds_and_loss(X, y, teacher_forcing=0.5)
        train_perplexity = torch.exp(loss)
        # logits shape : (n_timesteps, n_classes)
        self.train_acc(logits, y.squeeze(0))  # remove batch dimension
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_PPL', train_perplexity, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        logits, val_loss = self._get_preds_and_loss(X, y, teacher_forcing=0.0)

        val_perplexity = torch.exp(val_loss)
        self.val_acc(logits, y.squeeze(0))

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.log('val_PPL', val_perplexity, on_step=False, on_epoch=True, prog_bar=True)
        # return val_loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        # print(X[0][0])

        logits, test_loss = self._get_preds_and_loss(X, y, teacher_forcing=0.0)

        test_perplexity = torch.exp(test_loss)
        self.test_acc(logits, y.squeeze(0))  # remove the batch dimension

        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_PPL', test_perplexity, on_step=False, on_epoch=True, prog_bar=True)
        # return test_loss

    def _get_preds_and_loss(self, X, y, teacher_forcing):
        logits = self(X, y, teacher_forcing)
        logits = logits.squeeze(0)  # remove the batch dimension
        loss = self.loss_module(logits, y.squeeze(0))

        return logits, loss

# TODO: Implement attention mechanism
# class AttentionDecoder():
#     pass
