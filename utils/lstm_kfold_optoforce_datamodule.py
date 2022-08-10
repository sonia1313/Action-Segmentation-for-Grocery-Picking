"""
Author - Sonia Mathews
lstm_kfold_optoforce_datamodule.py

This script has been adapted by the author from:
https://github.com/Lightning-AI/lightning/blob/master/examples/pl_loops/kfold.py
to perfom KFold cross validation for the dataset used in this project.
(And to perform experiment tracking on Weights and Biases)
"""

import pytorch_lightning as pl
from pytorch_lightning.loops import FitLoop, Loop
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchmetrics import Accuracy, ConfusionMatrix
import os.path as osp

from utils.edit_distance import edit_score
from utils.metrics_utils import _get_average_metrics
from utils.optoforce_data_loader import OpToForceDataset
from utils.overlap_f1_metric import f1_score
from utils.plot_confusion_matrix import _plot_cm
from utils.tactile_preprocessing import *
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.model_selection import KFold

import wandb


#############################################################################################
#                           KFold Loop / Cross Validation Example                           #
# This example demonstrates how to leverage Lightning Loop Customization introduced in v1.5 #
# Learn more about the loop structure from the documentation:                               #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html                  #
#############################################################################################

#############################################################################################
#                           Step 1 / 5: Define KFold DataModule API                         #
# Our KFold DataModule requires to implement the `setup_folds` and `setup_fold_index`       #
# methods.                                                                                  #
#############################################################################################


class BaseKFoldDataModule(pl.LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int):
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int):
        pass


#############################################################################################
#                           Step 2 / 5: Implement the KFoldDataModule                       #
# The `KFoldDataModule` will take a train and test dataset.                                 #
# On `setup_folds`, folds will be created depending on the provided argument `num_folds`    #
# Our `setup_fold_index`, the provided train dataset will be split accordingly to        #
# the current fold split.                                                                   #
#############################################################################################

@dataclass
class OpToForceKFoldDataModule(BaseKFoldDataModule):
    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None

    def __init__(self, X_data: [float], y_data: [int], single: bool, clutter: bool, seed:int, batch_size: int = 1):
        self.batch_size = batch_size
        self.X_data = X_data
        self.y_data = y_data
        if single == True and clutter == False:
            train_size = 25
            test_size = 4

        elif single == False and clutter == True:
            train_size = 25
            test_size = 5
        else:  # when using single and clutter
            train_size = 55
            test_size = 4

        self.train_size = train_size  # 25 when using clutter
        self.test_size = test_size  # 5 when using clutter
        self.prepare_data_per_node = True
        self._log_hyperparams = True
        self.seed = seed

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == 'fit':
            dataset = OpToForceDataset(self.X_data, self.y_data)
            self.train_dataset, self.test_dataset = random_split(dataset, [self.train_size, self.test_size],
                                                                 generator=torch.Generator().manual_seed(self.seed))

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in
                       KFold(num_folds, shuffle=True, random_state=self.seed).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        # optoforce_train = DataLoader(self.optoforce_train, batch_size=1, shuffle=True)
        return DataLoader(self.train_fold)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset)

    def __post_init__(cls):
        super().__init__()


# to reduce variability, the trained models are ensembled and their predictions are
# averaged when estimating the model’s predictive performance on the test dataset.
class EnsembleVotingModel(pl.LightningModule):

    def __init__(self, model_cls: Type[pl.LightningModule], checkpoint_paths: List[str],
                 n_features: int, hidden_size: int, n_layers: int, dropout: float, lr: float, wb_project_name: str,
                 wb_group_name: str) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.models = torch.nn.ModuleList(
            [model_cls.load_from_checkpoint(p) for p in checkpoint_paths])

        self.acc = Accuracy(ignore_index=-1, multiclass=True)
        self.loss_module = nn.CrossEntropyLoss(ignore_index=-1)

        self.confusion_matrix = ConfusionMatrix(num_classes=6)
        self.counter = 0
        self.experiment_name = wb_group_name

        self.wb_ensemble = wandb.init(project=wb_project_name, group=self.experiment_name,
                                      job_type='test', dir='wandb_runs')

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        X, y = batch
        self.counter += 1

        # print(f"test_step_from ensemble: {X[0]}")

        logits_per_model = []

        for m in self.models:
            logits = self._get_preds(m, X)
            logits_per_model.append(logits)

        logits = torch.stack(logits_per_model).mean(0)

        y = y[0][:].view(-1)  # shape = [max_seq_len]

        loss = self.loss_module(logits, y)
        accuracy = self.acc(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds, targets = remove_padding(logits, y)

        cm = self.confusion_matrix(preds, targets)
        fig = _plot_cm(cm, path=f"ensemble_cm_figs/{self.experiment_name}-ensemble-cm-{self.counter}.png")

        f1_scores = f1_score(preds, targets)
        edit = edit_score(preds, targets)

        self.wb_ensemble.log({"test_loss": loss, "test_acc": accuracy, "ensemble_confusion_matrix": wandb.Image(fig)})
        return accuracy, edit, f1_scores

    def test_epoch_end(self, outputs):
        f1_10_mean, f1_25_mean, f1_50_mean, edit_mean, accuracy_mean = _get_average_metrics(outputs)

        print(f"average test f1 overlap @ 10%: {f1_10_mean}")
        print(f"average test f1 overlap @ 25%: {f1_25_mean}")
        print(f"average test f1 overlap @ 50%: {f1_50_mean}")
        print(f"average test edit: {edit_mean}")
        print(f"average test accuracy : {accuracy_mean}")

        self.wb_ensemble.log({"average_test_f1_10": f1_10_mean, "average_test_f1_25": f1_25_mean,
                              "average_test_f1_50": f1_50_mean, "average_test_edit": edit_mean,
                              "average_test_accuracy": accuracy_mean})

    def _get_preds(self, model, X):
        logits = model(X)
        # logits: max_seqlen-1,n_classes e.g.[194,6]

        logits = logits.type_as(X)
        return logits

    # def _get_average_metrics(self, outputs):
    #
    #     f1_10_outs = []
    #     f1_25_outs = []
    #     f1_50_outs = []
    #     edit_outs = []
    #     accuracy_outs = []
    #     for i, out in enumerate(outputs):
    #         a, e, f = out
    #         f1_10_outs.append(f[0])
    #         f1_25_outs.append(f[1])
    #         f1_50_outs.append(f[2])
    #
    #         edit_outs.append(e)
    #         accuracy_outs.append(a)
    #
    #     f1_10_mean = np.stack([x for x in f1_10_outs]).mean(0)
    #     f1_25_mean = np.stack([x for x in f1_25_outs]).mean(0)
    #     f1_50_mean = np.stack([x for x in f1_50_outs]).mean(0)
    #     edit_mean = np.stack([x for x in edit_outs]).mean(0)
    #     accuracy_mean = torch.mean(torch.stack([x for x in accuracy_outs]))
    #
    #     return f1_10_mean, f1_25_mean, f1_50_mean, edit_mean, accuracy_mean


#############################################################################################
#                           Step 4 / 5: Implement the  KFoldLoop                            #
# From Lightning v1.5, it is possible to implement your own loop. There is several steps    #
# to do so which are described in detail within the documentation                           #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html.                 #
# Here, we will implement an outer fit_loop. It means we will implement subclass the        #
# base Loop and wrap the current trainer `fit_loop`.                                        #
#############################################################################################


#############################################################################################
#                     Here is the `Pseudo Code` for the base Loop.                          #
# class Loop:                                                                               #
#                                                                                           #
#   def run(self, ...):                                                                     #
#       self.reset(...)                                                                     #
#       self.on_run_start(...)                                                              #
#                                                                                           #
#        while not self.done:                                                               #
#            self.on_advance_start(...)                                                     #
#            self.advance(...)                                                              #
#            self.on_advance_end(...)                                                       #
#                                                                                           #
#        return self.on_run_end(...)                                                        #
#############################################################################################


class KFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: str, n_features: int, hidden_size: int, n_layers: int,
                 dropout: float, lr: float, project_name:str, experiment_name:str, config:dict) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr

        #experiment tracking meta info
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config


    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")

        self.wb_run = wandb.init(reinit=True, project=self.project_name,
                                 group=self.experiment_name, job_type='cross-val',
                                 id=f'current_fold_{self.current_fold}',dir='wandb_runs', config=self.config)

        # tracking gradients and hyperparameters
        self.wb_run.watch(self.trainer.model, log='all', log_freq=1)

        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        # pass experiment tracking object
        # self.trainer.lightning_module.training_step(wbj_obj = self.wb_run)
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        self.wb_run.finish()
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)

        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module),
                                           checkpoint_paths
                                           , n_features=self.n_features,
                                           hidden_size=self.hidden_size,
                                           n_layers=self.n_layers,
                                           dropout=self.dropout,
                                           lr=self.lr,
                                           wb_project_name=self.project_name,
                                           wb_group_name=self.experiment_name)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()
        # print(voting_model.counter)

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
