# TODO: Read the image and clutter files
# TODO: Downsample to 3FS
# TODO: save in a pcikle
# TODO: Separate the images and labels for each seq
# TODO: pad sequences (!)
# TODO: remove padding (!)
# TODO: convert labels
# TODO: create tensors

import os
import pickle

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision import transforms

os.chdir("C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project")


def load_files(single=True, clutter=True):
    if single and clutter:
        with open('data/image/single_clutter_img_df.pkl', 'rb') as f:
            files = pickle.load(f)
    elif (single is True) and (clutter is False):
        with open('data/image/single_img_df.pkl', 'rb') as f:
            files = pickle.load(f)

    else:
        with open('data/image/clutter_img_df.pkl', 'rb') as f:
            files = pickle.load(f)

    return files


def _downsample(fps, files):
    """" 30 frames is approx 3FPS"""

    nth_frame = 30//fps
    data_df_seqs = []
    for file in files:
        data_df = pd.read_csv(file)
        data_df = data_df.iloc[::nth_frame,:]
        data_df_seqs.append(data_df)

    return data_df_seqs

def get_labels(sequences):
    actions_per_seq = []
    label_to_index_map = {'move-in': 0, 'manipulate': 1, 'grasp': 2, 'pick-up': 3, 'move-out': 4, 'drop': 5}
    for seq_df in sequences:
        actions = []
        for i in range(0, len(seq_df)):
            label = label_to_index_map[seq_df['label'].iloc[i]]
            actions.append(label)
        actions_per_seq.append(actions)


    return actions_per_seq









#in the data loader