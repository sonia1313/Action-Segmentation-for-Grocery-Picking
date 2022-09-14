""""
Author - Sonia Mathews
multimodal_get_dataset_utils.py

Script to create datasets composed of tensors, to then
pass to MultiModalDataset() class.
Improves efficiency during experimentation
"""
import pickle

from utils.image_preprocessing import get_labels, images_per_seq, get_tactile_tensors
from torch.nn.utils.rnn import pad_sequence
import torch
import os


def get_mm_dataset(single:bool, clutter:bool):
    os.chdir("TODO")

    if single:
        with open('data/multimodal/5fps/single_multi_modal_dataset_5fps.pkl','rb') as f:
            mm_dfs = pickle.load(f)
        print(len(mm_dfs))
    elif clutter:
        with open('data/multimodal/5fps/clutter_multi_modal_dataset_5fps.pkl','rb') as f:
            mm_dfs = pickle.load(f)
        print(len(mm_dfs))
    else:
        print("state either single(T/F) or clutter(T/F) only")

    labels_per_seq, fruits_per_seq, env_per_seq = get_labels(mm_dfs)

    padded_label_tensors = pad_sequence(labels_per_seq, batch_first=True, padding_value=-1)

    img_tensors_per_seq = images_per_seq(mm_dfs, downsample_img_size=32)

    padded_img_tensors = pad_sequence(img_tensors_per_seq, batch_first=True)

    tactile_tensors_per_seq = get_tactile_tensors(mm_dfs)

    padded_tactile_tensors = pad_sequence(tactile_tensors_per_seq, batch_first=True)

    return padded_img_tensors,padded_tactile_tensors,padded_label_tensors,fruits_per_seq,env_per_seq

def save_data():
    padded_img_tensors, padded_tactile_tensors, padded_label_tensors,fruits_per_seq, env_per_seq = get_mm_dataset(single=True,clutter=False)

    torch.save((padded_img_tensors, padded_tactile_tensors, padded_label_tensors,fruits_per_seq, env_per_seq),
               'data/multimodal/5fps/5fs_single_dataset_tensors.pt')

    padded_img_tensors,padded_tactile_tensors, padded_label_tensors, fruits_per_seq, env_per_seq = get_mm_dataset(single=False,clutter=True)

    torch.save((padded_img_tensors, padded_tactile_tensors,padded_label_tensors, fruits_per_seq, env_per_seq),
               'data/multimodal/5fps/5fs_clutter_dataset_tensors.pt')

if __name__ == '__main__':
    save_data()
