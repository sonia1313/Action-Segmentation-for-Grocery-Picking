
import os
import pickle

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision import transforms


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


def _downsample(files, fps=3):
    """" 30 frames is approx 1FPS"""

    nth_frame = 30 // fps
    data_df_seqs = []
    for file in files:
        data_df = file.iloc[::nth_frame, :]
        data_df_seqs.append(data_df)

    return data_df_seqs


def get_labels(sequences):
    actions_per_seq = []
    fruit_per_seq = []  # 1 for each seq
    env_per_seq = []  # 1 for each seq
    label_to_index_map = {'move-in': 0, 'manipulate': 1, 'grasp': 2, 'pick-up': 3, 'move-out': 4, 'drop': 5}
    fruit_to_index_map = {'avocado': 0, 'banana': 1, 'blueberry': 2}
    env_to_index_map = {'clutter': 0, 'single': 1}
    for seq_df in sequences:
        actions_in_seq = torch.zeros(size=(len(seq_df),), dtype=torch.long)
        for i in range(0, len(seq_df)):
            label = label_to_index_map[seq_df['label'].iloc[i]]
            actions_in_seq[i] = label
            if i == 0:
                fruit_label = fruit_to_index_map[seq_df['fruit'].iloc[i]]
                env_label = env_to_index_map[seq_df['environment'].iloc[i]]

                fruit_per_seq.append(torch.tensor(fruit_label, dtype=torch.long))
                env_per_seq.append(torch.tensor(env_label, dtype=torch.long))

        actions_per_seq.append(actions_in_seq)
    return actions_per_seq, fruit_per_seq, env_per_seq


def images_per_seq(sequences, downsample_img_size=32):
    # a list containing dataframe |t|path|single|
    imgs_tensors_per_seq = []  # EACH SEQ HAS SHAPE T X 3 X H X W

    transform = transforms.Compose([transforms.Resize((downsample_img_size, downsample_img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.48, 0.5, 0.48], [0.335, 0.335, 0.335])])

    for seq_df in sequences:
        imgs_in_seq = torch.zeros(size=(len(seq_df), 3, downsample_img_size, downsample_img_size))
        for i in range(0, len(seq_df)):
            img_pth = seq_df['path'].iloc[i]
            pth = img_pth.replace('\\', '/')
            pth = fr"{os.path.abspath(pth)}"
            pil_img = Image.open(pth)
            imgs_in_seq[i] = transform(pil_img)
        imgs_tensors_per_seq.append(imgs_in_seq)

    return imgs_tensors_per_seq


def remove_padding_img(predictions_padded, targets_padded):
    # predictions_padded_shape: B x T x n_classes
    # targets padded shape: B x T

    mask = (targets_padded >= 0).long()
    n = len([out for out in mask.squeeze() if out.all() >= 1])

    outputs = predictions_padded.squeeze()[:n, :]  # T x n_classes

    targets_padded = targets_padded.squeeze()
    targets = targets_padded[:n]

    return outputs, targets
