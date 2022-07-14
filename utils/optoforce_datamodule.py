import pytorch_lightning as pl
import torch
from pytorch_lightning.loops import FitLoop, Loop
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchmetrics import Accuracy
import os.path as osp
from torch.nn import functional as F
from utils.optoforce_data_loader import OpToForceDataset
from utils.preprocessing import *
from utils.remove_padding_and_one_hot import _remove_padding, _remove_one_hot
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from os import path
from typing import Any, Dict, List, Optional, Type
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.model_selection import KFold
from models.encoder_decoder_lstm import LitEncoderDecoderLSTM
PATH_TO_DIR = 'C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project'

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

    def __init__(self, X_data: [float], y_data: [int], batch_size: int = 1, train_size = 25, test_size = 5):
        self.batch_size = batch_size
        self.X_data = X_data
        self.y_data = y_data
        self.train_size = train_size #25 when using clutter
        self.test_size = test_size #5 when using clutter
        self.prepare_data_per_node = True
        self._log_hyperparams = True

    # def prepare_data(self) -> None:
    #     files, labels = get_files(PATH_TO_DIR)
    #
    #     frames, action_segment_td, ground_truth_actions = read_data(files, labels)
    #     frames = standardise_features(append_labels_per_frame(frames, action_segment_td, ground_truth_actions))
    #     actions_per_seq, unique_actions, index_label_map = one_hot_encode_labels(frames)
    #     self.X_data, self.y_data = pad_data(frames, actions_per_seq)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == 'fit':
            dataset = OpToForceDataset(self.X_data, self.y_data)
            self.train_dataset, self.test_dataset = random_split(dataset, [self.train_size, self.test_size], generator=torch.Generator().manual_seed(42))

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

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

    def __init__(self, model_cls: Type[pl.LightningModule], checkpoint_paths: List[str]) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.acc = Accuracy(ignore_index=-1)
        self.loss_module = nn.CrossEntropyLoss(ignore_index=-1)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:

        # Compute the averaged predictions over the `num_folds` models.
        X, y = batch

        #print(f"test_step_from ensemble: {X[0]}")

        logits_per_model = []

        for m in self.models:
            logits = self._get_preds(m,X,y) #not sure if this will work
            logits_per_model.append(logits)

        logits = torch.stack(logits_per_model).mean(0)


        loss = self.loss_module(logits,y.squeeze(0))
        self.acc(logits, y.squeeze(0))

        self.log("average_test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("average test_acc", self.acc, on_step=False, on_epoch=True, prog_bar=True)


    def _get_preds(self,model, X,y, teacher_forcing = 0.0):

        logits = model(X,y,teacher_forcing)
        logits = logits.squeeze(0) #remove the batch dimension
        return logits


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
    def __init__(self, num_folds: int, export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path

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
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

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


if __name__ == "__main__":
    pl.seed_everything(42)
    X_data, y_data, labels_map = preprocess_dataset(PATH_TO_DIR)
    model = LitEncoderDecoderLSTM()
    datamodule = OpToForceKFoldDataModule(X_data,y_data)
    trainer = pl.Trainer(
        max_epochs=10,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        #devices=2,
        accelerator="auto",
        #strategy="ddp", #gives AttributeError: 'DistributedDataParallel' object has no attribute 'test_step'
    )

    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = KFoldLoop(5, export_path="./")
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule)