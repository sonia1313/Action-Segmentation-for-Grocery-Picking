import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy


class EncoderLSTM(nn.Module):
    """ Encodes tactile time series data """

    def __init__(self, n_features, hidden_size=100, n_layers=2, n_classes=6):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes

        self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, batch_first=True)

    def forward(self, x, hidden):
        batch_size = x.shape[0]

        output, (h_n, c_n) = self.lstm(x, hidden)
        return output, (h_n,c_n)

    def _init_states(self, batch_size):
        if torch.cuda.is_available():
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True, device="cuda")
            # c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True, device="cuda")

        else:
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True)
            # c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True)

        return h0


class DecoderLSTM(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size=self.hidden_size, out_features=self.n_classes)
        self.softmax = nn.LogSoftmax(dim=1)

        return self.out, self.hidden_size

    def forward(self, x, hidden):
        """"
        hidden - the final hidden state from the encoder model is the context vector of the source sequence.
        x - is the target ouput
        """
        output, (h_n,c_n) = self.lstm(x, hidden)
        #output = self.softmax(self.linear(output[0]))
        output = self.linear(output[0]) #not sure
        return output, (h_n,c_n)

    def _init_states(self, batch_size):
        if torch.cuda.is_available():
            h0 = torch.zeros(1, 1, self.hidden_size, requires_grad=True, device="cuda")

        else:
            h0 = torch.zeros(1, 1, self.hidden_size, requires_grad=True)
        return h0

class EncoderDecoderLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderLSTM()
        self.decoder = DecoderLSTM()

    def forward(self,x, ):

        encoder_output, (encoder_hn, decoder_cn) = self.encoder(x)
        decoder_output, (decoder_hn, decoder_cn) = self.decoder(encoder_hn)

        return decoder_output


#encoder gives a context vector
#decoder uses context vector to predict h0, h1, h2...hn
#decoder can also use teacher forcing [h0,y0],[h1,y1],[h2,y2]

class LitEncoderDecoderLSTM(pl.LightningModule):
    def __init__(self, encoder_decoder):
        super().__init__()

        self.encoder_decoder = encoder_decoder
        self.loss_module = nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val = Accuracy()
        self.test = Accuracy()

    def training_step(self, batch, batch_idx):
        padded_X, padded_y = batch
        padded_logits = self.model(padded_X)
        logits, y = _remove_padding(padded_logits, padded_y)
        loss = self.loss_module(logits,y)

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        padded_X, padded_y = batch
        padded_logits = self.model(padded_X)
        logits, y = _remove_padding(padded_logits, padded_y)

        self.val_acc(logits, y)
        val_loss = self.loss_module(logits, y)

        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss


    def test_step(self, batch, batch_idx):
        padded_X, padded_y = batch
        padded_logits = self.model(padded_X)
        logits, y = _remove_padding(padded_logits, padded_y)

        #test_acc = (logits.argmax(dim=-1) == y).float().mean()

        test_loss = self.loss_module(logits, y)
        # test_loss = self.training_step(batch,batch_idx)

        self.test_acc(logits,y)

        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return test_loss



def _remove_padding(predictions_padded, targets_padded):
    mask = (targets_padded >= 0).long()
    n = len([out for out in mask.squeeze() if out.all() >= 1])
    outputs = predictions_padded[:n, :]

    targets_padded = targets_padded.squeeze()
    targets = targets_padded[:n, :]
    #_, targets = targets.max(dim=1)  # remove one hot encoding

    return outputs, targets


# TODO: Implement attention mechanism
# class AttentionDecoder():
#     pass
