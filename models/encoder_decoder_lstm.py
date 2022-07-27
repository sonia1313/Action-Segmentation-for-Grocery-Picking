import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix
import random
import wandb
from utils.plot_confusion_matrix import _plot_cm
from utils.preprocessing import remove_padding
from utils.overlap_f1_metric import f1_score
from utils.edit_distance import edit_score


class EncoderLSTM(nn.Module):
    """ Encodes tactile time series data """

    def __init__(self, n_features=6, hidden_size=100, n_layers=1, n_classes=6):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_classes = n_classes

        self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_size,
                            num_layers=self.n_layers, batch_first=True)

    def forward(self, x):
        batch_size = x.shape[0]

        h0, c0 = self._init_states(batch_size)
        output, (h_n, c_n) = self.lstm(x, (h0, c0))
        # print(f"encoder hidden shape {h_n.shape}")
        # print(f"encoder cell shape {c_n.shape}")
        #
        # print(f"output shape {output.shape}")
        return h_n, c_n

    def _init_states(self, batch_size):
        if torch.cuda.is_available():
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True, device="cuda")
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True, device="cuda")

        else:
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True)

        return h0, c0


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
        #
        # print(x)
        # x = torch.LongTensor(x.view(1,1,1))

        # print(f"decoder input shape: {x.shape}")
        # lstm input size: (seq_len, batch,n_features) = (1,1,1)
        # print(f"decoder input: {x}")
        # print(f"decoder hidden input shape: {hidden.shape}")
        # print(f"decoder cell input shape : {cell.shape}")
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        # output shape: (1,1,100) #sequence_legth, batch_size, hidden

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
    def __init__(self, n_features, hidden_size, n_layers):
        super().__init__()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # assert n_features == 6
        self.encoder = EncoderLSTM(n_features, hidden_size, n_layers)
        self.decoder = DecoderLSTM(hidden_size)

        assert self.encoder.hidden_size == self.decoder.hidden_size

    def forward(self, x, y, teacher_forcing_ratio, n_classes=6):
        batch_size = y.shape[0]
        optoforce_seq_len = y.shape[1]

        outputs = torch.zeros(batch_size, optoforce_seq_len, n_classes)

        hidden, cell = self.encoder(x)
        # print("context vector shape")
        # print(hidden.shape)
        # print("beginning decoder input")
        # print(decoder_input.shape)
        decoder_input = y[0][0]
        # print("first label to decoder input")
        # print(decoder_input) #takes the first label as input
        decoder_input = decoder_input.type(torch.float32)
        decoder_input = decoder_input.view(1, 1, 1)
        for t in range(1, optoforce_seq_len):
            # print(t)

            # print(f"current t {t}")
            # print(f"decoder input shape at time {t} = {decoder_input.shape}")
            # print(f"h_x.shape {hidden.shape}")
            # print(f"cell.shape {cell.shape}")
            # decoder_input (1,1,1) bactch_size,seq,len,input
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # output shape (1,6)

            # print(f"output predicted at time {t} = {output}")
            outputs[0][t] = output  # instead of t -1

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
    def __init__(self, n_features, hidden_size, n_layers, experiment_tracking):
        super().__init__()

        # device = torch.device("cuda:0" if torch.cu da.is_available() else "cpu")
        self.encoder_decoder_model = EncoderDecoderLSTM(n_features, hidden_size, n_layers)
        self.loss_module = nn.CrossEntropyLoss(ignore_index=-1)
        self.train_acc = Accuracy(ignore_index=-1, multiclass=True)
        self.val_acc = Accuracy(ignore_index=-1, multiclass=True)
        self.test_acc = Accuracy(ignore_index=-1, multiclass=True)

        self.experiment_tracking = experiment_tracking
        self.test_counter = 0
        self.val_counter = 0
        self.confusion_matrix = ConfusionMatrix(num_classes=6)
        # self.exp_name = 'testing_in_gpu'

    def forward(self, X, y, teacher_forcing):
        logits = self.encoder_decoder_model(X, y, teacher_forcing)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch
        # print(X.shape)
        # print(y.shape)
        logits, loss = self._get_preds_and_loss(X, y, teacher_forcing=0.5)
        y = y[0][1:].view(-1)  # shape = [max_seq_len-1]
        accuracy = self.train_acc(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding(logits, y)

        f1_scores = f1_score(preds, targets)
        edit = edit_score(preds, targets)

        if self.experiment_tracking:
            wandb.log({"epoch": self.current_epoch, "train_loss": loss, "train_accuracy": accuracy,
                       "f1_overlap_10": f1_scores[0], "f1_overlap_25": f1_scores[1],
                       "f1_overlap_50": f1_scores[2], "edit_score": edit})

        # cm = self.confusion_matrix(preds, targets)

        # _ = _plot_cm(cm=cm, path=f"training_confusion_matrix/{self.exp_name}_{self.test_counter}.png")

        # print(f"accuracy caluclated by torch metric: {accuracy}")

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        self.val_counter += 1
        print(self.val_counter)
        logits, val_loss = self._get_preds_and_loss(X, y, teacher_forcing=0.0)
        y = y[0][1:].view(-1)  # shape = [max_seq_len-1]

        accuracy = self.val_acc(logits, y.squeeze(0))

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding(logits, y)
        cm = self.confusion_matrix(preds, targets)

        f1_scores = f1_score(preds, targets)
        print(f1_scores)
        edit = edit_score(preds, targets)

        # TODO: convert to a plotty fig to save on wb
        cm_plot = _plot_cm(cm, path=f"confusion_matrix_figs/{self.val_counter}_val.png")

        if self.experiment_tracking:
            wandb.log({"epoch": self.current_epoch, "val_loss": val_loss, "val_accuracy": accuracy,
                       "val_f1_overlap_10": float(f1_scores[0]), "val_f1_overlap_25": float(f1_scores[1]),
                       "val_f1_overlap_50": float(f1_scores[2]), "val_edit_score": edit})

        # return {"val_accuracy": accuracy, "val_edit": edit_score, "val_f1_scores": f1_scores}
        return accuracy, edit, f1_scores

    def validation_epoch_end(self, outputs):

        f1_10_mean, f1_25_mean, f1_50_mean, edit_mean, accuracy_mean = self._get_average_metrics(outputs)

        if self.experiment_tracking:
            wandb.log({"average_val_f1_10": f1_10_mean, "average_val_f1_25": f1_25_mean,
                       "average_val_f1_50": f1_50_mean, "average_val_edit": edit_mean,
                       "average_val_accuracy": accuracy_mean})

    def test_step(self, batch, batch_idx):

        X, y = batch
        # print(X[0][0])
        self.test_counter += 1
        print(f"test:{self.test_counter}")

        logits, test_loss = self._get_preds_and_loss(X, y, teacher_forcing=0.0)
        y = y[0][1:].view(-1)  # shape = [max_seq_len-1]

        accuracy = self.test_acc(logits, y.squeeze(0))  # remove the batch dimension

        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding(logits, y)
        cm = self.confusion_matrix(preds, targets)

        f1_scores = f1_score(preds, targets)
        print(f1_scores)
        edit = edit_score(preds, targets)

        cm_plot = _plot_cm(cm, path=f"confusion_matrix_figs/{self.test_counter}_test.png")

        if self.experiment_tracking:
            wandb.log({"test_loss": test_loss, "test_accuracy": accuracy})

        return accuracy, edit, f1_scores

    def test_epoch_end(self, outputs):

        f1_10_mean, f1_25_mean, f1_50_mean, edit_mean, accuracy_mean = self._get_average_metrics(outputs)

        print(f"average test f1 overlap @ 10%: {f1_10_mean}")
        print(f"average test f1 overlap @ 25%: {f1_25_mean}")
        print(f"average test f1 overlap @ 50%: {f1_50_mean}")
        print(f"average test edit: {edit_mean}")
        print(f"average test accuracy : {accuracy_mean}")

        if self.experiment_tracking:
            wandb.log({"average_test_f1_10": f1_10_mean, "average_test_f1_25": f1_25_mean,
                       "average_test_f1_50": f1_50_mean, "average_test_edit": edit_mean,
                       "average_test_accuracy": accuracy_mean})

    def _get_average_metrics(self,outputs):

        f1_10_outs = []
        f1_25_outs = []
        f1_50_outs = []
        edit_outs = []
        accuracy_outs = []
        for i, out in enumerate(outputs):
            a, e, f = out
            f1_10_outs.append(f[0])
            f1_25_outs.append(f[1])
            f1_50_outs.append(f[2])

            edit_outs.append(e)
            accuracy_outs.append(a)

        f1_10_mean = np.stack([x for x in f1_10_outs]).mean(0)
        f1_25_mean = np.stack([x for x in f1_25_outs]).mean(0)
        f1_50_mean = np.stack([x for x in f1_50_outs]).mean(0)
        edit_mean = np.stack([x for x in edit_outs]).mean(0)
        accuracy_mean = torch.mean(torch.stack([x for x in accuracy_outs]))

        return f1_10_mean, f1_25_mean, f1_50_mean, edit_mean, accuracy_mean


    def _get_preds_and_loss(self, X, y, teacher_forcing):

        logits = self(X, y, teacher_forcing)

        # logits: batch_size,max_seqlen-1,n_classes e.g.[1,194,6]
        # y: batch_size,max_seqlen-1 e.g. [1,194]
        logits_dim = logits.shape[-1]

        logits = logits[0][1:].view(-1, logits_dim)
        y = y[0][1:].view(-1)
        # logits: max_seqlen-1,n_classes e.g.[193,6]
        # y: max_seqlen-1 e.g. [193]

        logits = logits.type_as(X)

        loss = self.loss_module(logits, y)

        return logits, loss

# TODO: Implement attention mechanism
# class AttentionDecoder():
#     pass
