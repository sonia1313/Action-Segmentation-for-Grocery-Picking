
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch
from action_segmentation_module import ActionSegmentationModule
from utils.optoforce_data_loader import load_data
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import *
#from pytorch_lightning.loggers import WandbLogger



def _preprocess_dataset(PATH_TO_DIR):
  files, labels = get_files(PATH_TO_DIR)

  frames, action_segment_td, ground_truth_actions = read_data(files, labels)
  frames = standardise_features(append_labels_per_frame(frames, action_segment_td, ground_truth_actions))
  actions_per_seq, unique_actions, index_label_map = one_hot_encode_labels(frames)
  X_data, y_data = pad_data(frames,actions_per_seq)

  return X_data, y_data
#
# class OpToForceDataset(Dataset):
#     def __init__(self, sequences, actions):
#         self.X = sequences
#         self.y = actions
#
#     def __len__(self):
#         return self.X.shape[0]
#
#     def __getitem__(self, item):
#         return self.X[item], self.y[item]
#
#
# def load_data(x_data, y_data):
#     train_dataset = OpToForceDataset(x_data, y_data)
#     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#
#     return train_loader


def main():
    #wandb.login()
    COLAB_GPU = 'COLAB_GPU' in os.environ
    PATH_TO_DIR = 'C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project'
    n_gpu = 0

    if COLAB_GPU:
        PATH_TO_DIR = '/content/drive/Othercomputers/Dell/Action-Segmentation-Project'
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(torch.cuda.get_device_properties(device).name)
            n_gpu = 1

    pl.seed_everything(42, workers=True)
    #checkpoint_callback = ModelCheckpoint(dirpath='//content/drive/Othercomputers/Dell/Action-Segmentation-Project/lightning_logs', save_top_k=2, monitor="val_loss")
    #logger = WandbLogger(project = "Action_Segmentation", log_model='all')
    trainer = pl.Trainer(default_root_dir=f"{PATH_TO_DIR}/checkpoints",  gpus=n_gpu, max_epochs=10, deterministic=True,progress_bar_refresh_rate=20)

    model = ActionSegmentationModule()

    # optoforce_data = OpToForceDataModule(PATH_TO_DIR=PATH_TO_DIR)
    # optoforce_data.prepare_data()
    # optoforce_data.setup()
    X_data, y_data = _preprocess_dataset(PATH_TO_DIR)
    train_loader, val_loader, test_loader = load_data(X_data,y_data)
    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    main()
