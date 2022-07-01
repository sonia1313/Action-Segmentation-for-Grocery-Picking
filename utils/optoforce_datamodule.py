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

PATH_TO_DIR = 'C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project'


class BaseKFoldDataModule(pl.LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int):
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int):
        pass


@dataclass
class OpToForceDataModule(BaseKFoldDataModule):
    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None

    def __init__(self, X_data: [float], y_data: [int], batch_size: int = 1):
        self.batch_size = batch_size
        self.X_data = X_data
        self.y_data = y_data
        self.train_size = int(0.8 * len(X_data))
        self.test_size = int(0.2 * len(X_data))
        self.prepare_data_per_node = True
        self._log_hyperparams = True

    # def prepare_data(self) -> None:
    #     files, labels = get_files(PATH_TO_DIR)
    #
    #     frames, action_segment_td, ground_truth_actions = read_data(files, labels)
    #     frames = standardise_features(append_labels_per_frame(frames, action_segment_td, ground_truth_actions))
    #     actions_per_seq, unique_actions, index_label_map = one_hot_encode_labels(frames)
    #     self.X_data, self.y_data = pad_data(frames, actions_per_seq)

    def setup(self, stage=None):
        if stage is None or stage == 'fit':
            dataset = OpToForceDataset(self.X_data, self.y_data)
            self.train_dataset, self.test_dataset = random_split(dataset, [self.train_size, self.test_size])

    def setup_folds(self, num_folds: int):
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int):
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self):
        # optoforce_train = DataLoader(self.optoforce_train, batch_size=1, shuffle=True)
        return DataLoader(self.train_fold)

    def val_dataloader(self):
        return DataLoader(self.val_fold)

    def test_dataloader(self):
        return DataLoader(self.test_dataset)

    def __post_init__(cls):
        super().__init__()


class EnsembleVotingModel(pl.LightningModule):

    def __init__(self, model_cls: Type[pl.LightningModule], checkpoint_paths: List[str]) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.acc = Accuracy()
        self.loss_module = nn.CrossEntropyLoss()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:

        # Compute the averaged predictions over the `num_folds` models.
        logits = torch.stack([_remove_padding(m(batch[0]),batch[1]) for m in self.models]).mean(0)

        targets = _remove_one_hot(batch[1])

        loss = self.loss_module(logits,targets)
        self.acc(logits, targets)
        self.log("average test_acc", self.acc)
        self.log("average_test_loss", loss)


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
