import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.optoforce_data_loader import OpToForceDataset
from utils.preprocessing import *

class OpToForceDataModule(pl.LightningDataModule):
    def __init__(self, PATH_TO_DIR,batch_size = 1):
        self.batch_size = 1
        self.PATH_TO_DIR = PATH_TO_DIR

    def prepare_data(self):
        files, labels = get_files(self.PATH_TO_DIR)

        frames, action_segment_td, ground_truth_actions = read_data(files, labels)
        frames = standardise_features(append_labels_per_frame(frames, action_segment_td, ground_truth_actions))
        actions_per_seq, unique_actions, index_label_map = one_hot_encode_labels(frames)
        self.X_data, self.y_data = pad_data(frames,actions_per_seq)

    def setup(self,stage = None):
        if stage is None or stage == 'fit':
          self.optoforce_train = OpToForceDataset(self.X_data,self.y_data)

    def train_dataloader(self):
        optoforce_train = DataLoader(self.optoforce_train, batch_size=1, shuffle=True)
        return optoforce_train


