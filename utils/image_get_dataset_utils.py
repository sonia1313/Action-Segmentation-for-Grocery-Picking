""""
Script to create datasets composed of tensors, to then
pass to ImageDataset() class.
Improves efficiency during experimentation
"""

from utils.image_preprocessing import get_labels, images_per_seq
from torch.nn.utils.rnn import pad_sequence
import torch
import os


def get_image_dataset(single:bool, clutter:bool):
    os.chdir("TODO")

    if single:
        img_dfs = torch.load('data/image/5fps/single_dataset_5fps.pt')
        print(len(img_dfs))
    elif clutter:
        img_dfs = torch.load('data/image/5fps/clutter_dataset_5fps.pt')
        print(len(img_dfs))
    else:
        print("state either single(T/F) or clutter(T/F) only")

    labels_per_seq, fruits_per_seq, env_per_seq = get_labels(img_dfs)

    padded_label_tensors = pad_sequence(labels_per_seq, batch_first=True, padding_value=-1)

    img_tensors_per_seq = images_per_seq(img_dfs, downsample_img_size=32)

    padded_img_tensors = pad_sequence(img_tensors_per_seq, batch_first=True)

    return padded_img_tensors,padded_label_tensors,fruits_per_seq,env_per_seq

def save_data():
    padded_img_tensors, padded_label_tensors, fruits_per_seq, env_per_seq = get_image_dataset(single=True,clutter=False)

    torch.save((padded_img_tensors, padded_label_tensors, fruits_per_seq, env_per_seq),
               'data/image/5fps/5fs_single_dataset_tensors.pt')

    padded_img_tensors, padded_label_tensors, fruits_per_seq, env_per_seq = get_image_dataset(single=False,clutter=True)

    torch.save((padded_img_tensors, padded_label_tensors, fruits_per_seq, env_per_seq),
               'data/image/5fps/5fs_clutter_dataset_tensors.pt')

if __name__ == '__main__':
    save_data()
