"""
Author - Sonia Mathews
conv_lstm.py


"""

import torch
import wandb
from torch import nn
from torchmetrics import Accuracy, ConfusionMatrix
import pytorch_lightning as pl
from utils.edit_distance import edit_score
from utils.metrics_utils import _get_average_metrics
from utils.overlap_f1_metric import f1_score
from utils.plot_confusion_matrix import _plot_cm
from utils.tactile_preprocessing import remove_padding


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        :param input_dim: int
            Number of channels in input image
        :param hidden_dim: int
            Number of channels in hidden state
        :param kernel_size:
            Size of filter used for convolution operation
        :param bias:bool
            If true, adds a learnable bias to the output. Default: True
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = self.kernel_size // 2

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              stride=1,
                              bias=self.bias)

    def forward(self, input_tensor, current_states):
        """

        :param input_tensor: tensor
        shape: batch_size (B) x num_channels (C) x height (H) x width (W)
        :param current_states: tuple
            tuple containing hidden state and cell_state tensor
        :return:new_hidden_state, new_cell_state
        """

        curr_hidden_state, curr_cell_state = current_states
        # curr_hidden_state shape: B x hidden_dim x H x W

        # curr_cell_state shape: B x hidden_dim x H x W

        # input_tensor shape: B x C x H x W when n_layer = 0 else B x hidden_dim x hidden_dim x W

        conv_input = torch.cat([input_tensor, curr_hidden_state], dim=1)

        combined_conv_gates = self.conv(conv_input)

        forget_gate, input_gate, output_gate, cell_gate = combined_conv_gates.chunk(4, 1)

        forget_gate = torch.sigmoid(forget_gate)
        input_gate = torch.sigmoid(input_gate)
        output_gate = torch.sigmoid(output_gate)
        cell_gate = torch.tanh(cell_gate)

        new_cell_state = (forget_gate * curr_cell_state) + (input_gate * cell_gate)
        new_hidden_state = output_gate * torch.tanh(new_cell_state)

        return new_hidden_state, new_cell_state


class ManyToManyConvLSTM(nn.Module):
    def __init__(self, img_size, input_dim, hidden_dim, n_layers, kernel_size, pool_size, bias=True, n_classes=6):
        """

        :param img_size: int
            H and W dims are assumed to be equal
        :param input_dim: int
            no channels in input image
        :param hidden_dim: int
        :param n_layers:int
        :param kernel_size: int or tuple
        :param pool_size: int or tuple
        :param bias: bool
        :param n_classes: int
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.bias = bias
        self.n_classes = n_classes
        self.img_size = img_size  # assumes height and width are the same
        self.pool_size = pool_size

        conv_lstm_per_layer = []
        for i in range(n_layers):
            in_channels = self.input_dim if i == 0 else self.hidden_dim

            conv_lstm_per_layer.append(ConvLSTMCell(input_dim=in_channels,
                                                    hidden_dim=self.hidden_dim,
                                                    kernel_size=self.kernel_size,
                                                    bias=self.bias))

        self.conv_lstm_per_layer = nn.ModuleList(conv_lstm_per_layer)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=self.pool_size)
        # INPUT_IMG_SIZE = 25 X 25

        downsample_size = ((self.img_size - self.pool_size) // self.pool_size) + 1  # [(W-K+2P)/S]+1
        in_features = (downsample_size ** 2) * self.hidden_dim

        self.linear1 = nn.Linear(in_features=in_features, out_features=500)
        self.linear2 = nn.Linear(in_features=500, out_features=50)
        self.linear3 = nn.Linear(in_features=50, out_features=self.n_classes)

    def forward(self, input):
        # shape of input: batch_size (B), seq_len(L), n_channels(C), height(H), width(W)
        input_seq_length = input.shape[1]
        batch_size = input.shape[0]
        height = input.shape[3]
        width = input.shape[4]
        h_0, c_0 = self._init_states(batch_size=batch_size, height=height, width=width)

        conv_lstm_in = input
        for i in range(self.n_layers):
            h_i, c_i = h_0[i], c_0[i]
            hidden_states = []
            # print(hidden_states.shape)
            for t in range(input_seq_length):
                h_i, c_i = self.conv_lstm_per_layer[i](conv_lstm_in[:, t, :, :, :], (h_i, c_i))
                hidden_states.append(h_i)

            layer_hid_out = torch.stack(hidden_states, dim=1)
            # layer_hid_out = hidden_states
            conv_lstm_in = layer_hid_out

            if i == self.n_layers - 1:
                out = layer_hid_out  # hidden states for every timestep from last layer
                h_n = h_i  # hidden state at last timestep at last layer
                c_n = c_i  # cell states at last timestep at last layer

        # out shape: B x L x hidden_dim x H x W

        out = self.pool(self.relu(out.squeeze(0)))  # remove batch dimension

        # out shape: L x hidden_dim x H x W
        out_flatten = out.flatten(1)
        # out_flatten shape: L x hidden_dim

        x = self.linear1(out_flatten)
        # x shape: L x 500
        x = self.linear2(x)
        # x shape: L x 50
        logits = self.linear3(x)
        # logits shape : L x n_classes
        return logits, (out, h_n, c_n)

    def _init_states(self, batch_size, height, width):
        if torch.cuda.is_available():
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, height, width, requires_grad=True,
                             device="cuda")
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, height, width, requires_grad=True,
                             device="cuda")

        else:
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, height, width, requires_grad=True)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, height, width, requires_grad=True)

        return h0, c0


class LitManyToManyConvLSTM(pl.LightningModule):

    def __init__(self, img_size, input_dim, hidden_dim, n_layers, kernel_size, pool_size, exp_name, lr,
                 experiment_tracking=True):
        super().__init__()
        self.save_hyperparameters()
        self.conv_lstm = ManyToManyConvLSTM(img_size=img_size, input_dim=input_dim, hidden_dim=hidden_dim,
                                            n_layers=n_layers,
                                            kernel_size=kernel_size, pool_size=pool_size)
        self.lr = lr

        self.loss_module = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(multiclass=True)
        self.val_acc = Accuracy(multiclass=True)
        self.test_acc = Accuracy(multiclass=True)

        self.experiment_tracking = experiment_tracking
        self.test_counter = 0
        self.val_counter = 0
        self.confusion_matrix = ConfusionMatrix(num_classes=6)
        self.experiment_name = exp_name

    def forward(self, X):
        logits,_ = self.conv_lstm(X)  # seq_len x n_classes

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch

        #X shape: B x L x C x H X W
        #y shape: B x L

        logits = self(X)  # shape = [max_seq_len] X 6
        #y = y[0][:].view(-1)  # shape = [max_seq_len]
        y = y.squeeze(0)
        loss = self.loss_module(logits, y)
        accuracy = self.train_acc(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # TODO:check padding
        #preds, targets = remove_padding(logits, y)

        f1_scores = f1_score(logits, y)
        edit = edit_score(logits, y)

        cm = self.confusion_matrix(logits, y)

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

        logits = self(X)  # shape = [max_seq_len] X 6
        #y = y[0][:].view(-1)  # shape = [max_seq_len]
        y = y.squeeze(0)

        val_loss = self.loss_module(logits, y)
        accuracy = self.val_acc(logits, y)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        #preds, targets = remove_padding(logits, y)
        cm = self.confusion_matrix(logits, y)

        f1_scores = f1_score(logits, y)
        # print(f1_scores)
        edit = edit_score(logits, y)

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

        logits = self(X)  # shape = [max_seq_len] X 6
        #y = y[0][:].view(-1)  # shape = [max_seq_len]
        y = y.squeeze(0)
        test_loss = self.loss_module(logits, y)

        accuracy = self.test_acc(logits, y)

        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        #preds, targets = remove_padding(logits, y)

        cm = self.confusion_matrix(logits, y)

        f1_scores = f1_score(logits, y)
        # print(f1_scores)
        edit = edit_score(logits, y)

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
