"""
Author - Sonia Mathews
cnn_tcn.py
"""
import torch
import torch.nn as nn
import wandb
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix
from models.tcn import TemporalConvNet
from utils.edit_distance import edit_score
from utils.image_preprocessing import remove_padding_img
from utils.metrics_utils import _get_average_metrics
from utils.overlap_f1_metric import f1_score
from utils.plot_confusion_matrix import _plot_cm


class CNN(nn.Module):
    def __init__(self, input_channels=3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=self.kernel_size,
                               padding=self.padding)
        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.kernel_size, padding=self.padding)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, padding=self.padding)

        # self.fc1 = nn.Linear(in_features=32*4*4,out_features=10)

    def forward(self, x):
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))

        x = self.max_pool(self.relu(self.conv3(x)))  # x out shape: L x C x H X W

        # print(x.shape)
        x = x.flatten(start_dim=1)
        # print(x.shape)
        # out = self.fc1(x)
        # print(x.shape)
        return x


class CNN_TCN(nn.Module):
    def __init__(self, tcn_dropout, cnn_input_channels=3, cnn_kernel_size=3, tcn_hid=50,
                 tcn_levels=2, tcn_kernel_size=8, n_classes=6):
        super().__init__()
        self.cnn = CNN(input_channels=cnn_input_channels, kernel_size=cnn_kernel_size)  # input: B*LxCxHxW output:LxH
        self.num_channels = [tcn_hid] * tcn_levels
        self.tcn = TemporalConvNet(n_features=512, n_classes=6,  # in features is from final conv layer from cnn: L x
                                   # 32 * 4 * 4
                                   num_channels_per_level=self.num_channels,
                                   kernel_size=tcn_kernel_size, stride=1,
                                   dropout=tcn_dropout)  # INPUT: BXCXL output: L x n_classes

    def forward(self, x):
        cnn_outs = self.cnn(x)
        # print("output of cnn shape")
        # print(cnn_outs.shape)
        tcn_in = cnn_outs.unsqueeze(0)
        tcn_in = torch.permute(tcn_in, (0, 2, 1))
        # print("input to tcn shape")
        # print(tcn_in.shape)
        logits, hidden = self.tcn(tcn_in)  # logits shape L x n_classes
        # print("logits shape")
        # print(logits.shape)
        return logits, hidden


class LitCNN_TCN(pl.LightningModule):
    def __init__(self, exp_name, lr, tcn_dropout, cnn_input_channels=3, cnn_kernel_size=3, tcn_nhid=50,
                 tcn_levels=2, tcn_kernel_size=8,
                 experiment_tracking=True, n_classes=6):
        super().__init__()
        self.cnn_tcn = CNN_TCN(cnn_input_channels=cnn_input_channels,
                               cnn_kernel_size=cnn_kernel_size,
                               tcn_levels=tcn_levels,
                               tcn_hid=tcn_nhid,
                               tcn_kernel_size=tcn_kernel_size,
                               tcn_dropout=tcn_dropout)
        self.save_hyperparameters()
        self.n_classes = n_classes
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

    def forward(self, x):

        logits, _ = self.cnn_tcn(x)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, batch, batch_idx):
        X, y, fruit = batch

        c_in = X.flatten(start_dim=0, end_dim=1)

        logits = self(c_in)

        y = y.squeeze(0)
        loss = self.loss_module(logits, y)

        accuracy = self.train_acc(logits, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding_img(logits, y)
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
        X, y, fruit = batch

        c_in = X.flatten(start_dim=0, end_dim=1)

        logits = self(c_in)

        # y = y[0][:].view(-1)
        y = y.squeeze(0)
        loss = self.loss_module(logits, y)

        accuracy = self.val_acc(logits, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding_img(logits, y)
        cm = self.confusion_matrix(preds, targets)

        f1_scores = f1_score(preds, targets)
        # print(f1_scores)
        edit = edit_score(preds, targets)

        fig = _plot_cm(cm, path=f"confusion_matrix_figs/{self.experiment_name}-validation-cm-{self.val_counter}.png")

        if self.experiment_tracking:
            wandb.log({"epoch": self.current_epoch, "val_loss": loss, "val_accuracy": accuracy,
                       "val_f1_overlap_10": float(f1_scores[0]), "val_f1_overlap_25": float(f1_scores[1]),
                       "val_f1_overlap_50": float(f1_scores[2]), "val_edit_score": edit})
            if self.current_epoch % 5 == 0:
                wandb.log({"validation_confusion_matrix": wandb.Image(fig)})
        return accuracy, edit, f1_scores

    def validation_epoch_end(self, outputs):

        f1_10_mean, f1_25_mean, f1_50_mean, edit_mean, accuracy_mean = _get_average_metrics(outputs)

        if self.experiment_tracking:
            wandb.log({"average_val_f1_10": f1_10_mean, "average_val_f1_25": f1_25_mean,
                       "average_val_f1_50": f1_50_mean, "average_val_edit": edit_mean,
                       "average_val_accuracy": accuracy_mean})

    def test_step(self, batch, batch_idx):
        X, y, fruit = batch

        c_in = X.flatten(start_dim=0, end_dim=1)
        logits = self(c_in)

        # y = y[0][:].view(-1)
        y = y.squeeze(0)
        loss = self.loss_module(logits, y)

        accuracy = self.test_acc(logits, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding_img(logits, y)
        # print(f"preds shape: {preds.shape}")
        # print(f"target shape: {targets.shape}")
        cm = self.confusion_matrix(preds, targets)

        f1_scores = f1_score(preds, targets)
        # print(f1_scores)
        edit = edit_score(preds, targets)

        fig = _plot_cm(cm, path=f"confusion_matrix_figs/{self.experiment_name}-test-{self.test_counter}.png")

        if self.experiment_tracking:
            wandb.log({"test_loss": loss, "test_accuracy": accuracy, "test_confusion_matrix": wandb.Image(fig)})

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

# def main():
#     # model = CNN()
#     # batch_size = 1
#     # time_steps = 100
#     # n_channels = 3
#     # size = 32
#     # dummy_x = torch.rand(batch_size,time_steps,n_channels,size,size)
#     # c_in = dummy_x.view(1*100,3,32,32)
#     # print(c_in.shape)
#     # x = model(c_in)
#     # print(x.shape)
#
#     # ----testing-cnn-tcn
#
#     # model = LitCNN_TCN(exp_name=None,lr=0.0,tcn_dropout=0.0)
#     # batch_size = 1
#     # time_steps = 100
#     # n_channels = 3
#     # size = 32
#     # dummy_x = torch.rand(batch_size, time_steps, n_channels, size, size)
#     # c_in = dummy_x.flatten(start_dim=0, end_dim=1)
#     # print("input to cnn")
#     # print(c_in.shape)
#     # logits = model(c_in)
#     # print(logits.shape)
#
#
#
# # if __name__ == '__main__':
# #     main()
