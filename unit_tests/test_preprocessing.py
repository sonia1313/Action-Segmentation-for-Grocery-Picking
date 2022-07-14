import random
import re

import torch

from utils.preprocessing import *
import numpy as np

import pandas as pd
import datatest as dt


def test_get_files():
    X, y = get_files('C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project')
    #
    # for i,j in zip(X,y):
    #     print(i)
    #     print(j)
    # #print(y)

    fruits = 'avocado|banana|blueberry'
    env = 'clutter|single'
    print()
    fruit_and_env = re.findall('avocado|banana|blueberry|clutter|single',X[0])
    print(fruit_and_env)
    print(X[0])
    assert len(X) == 30
    assert len(y) == 30




def test_read_data():
    X, y = get_files('C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project')
    frames, action_segment_td, ground_truth_actions = read_data(X, y)

    assert len(frames[0].columns) == 7
    # assert np.all(frames[0].columns == ['time', 'index', 'middle','thumb','label'])
    dt.validate(frames[0].columns, {'time', 'index', 'middle', 'thumb','fruit','environment','label'})
    assert len(action_segment_td) == len(ground_truth_actions)
    dt.validate(frames[0].label, str)
    dt.validate(frames[0].fruit, str)
    dt.validate(frames[0].environment, str)
    print()
    print()
    print(frames[0].head())


def test_encode_labels():
    X, y = get_files('C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project')
    frames, action_segment_td, ground_truth_actions = read_data(X, y)
    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)
    frames = standardise_features(frames)

    actions_per_seq, label_to_index_map = encode_labels(frames)
    print()
    #print(actions_per_seq[0])
    #assert len(unique_actions) == 6
    dt.validate(actions_per_seq[0], int)
    random_int = random.randint(0,10)
    for i in range(0,30):
        print(i)
        assert len(frames[i] == len(actions_per_seq[i]))
    indices = set(label_to_index_map)
    print(indices)
    for i in range(0,30):
        for encoded_label, label in zip(actions_per_seq[i], frames[i].label):
            assert label_to_index_map[label] == encoded_label

    for encoded_label, label in zip(actions_per_seq[random_int], frames[random_int].label):
        print(f"{label} and {encoded_label}")



def test_pad_data():
    X, y = get_files('C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project')
    frames, action_segment_td, ground_truth_actions = read_data(X, y)
    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)
    frames = standardise_features(frames)

    max_len = max([len(f) for f in frames])
    print(max_len)
    actions_per_seq, label_to_index_map = encode_labels(frames)
    # print(label_to_index_map)
    padded_numeric_features, padded_labels = pad_data(frames, actions_per_seq)

    assert len(padded_numeric_features[0]) == len(padded_labels[0])
    print()
    print(label_to_index_map)
    print(padded_labels[0])

    assert (len(padded_labels[np.random.randint(0, 30)]) == len(padded_labels[np.random.randint(0, 30)]))

    print(f"max of sequence length {len(padded_labels[0])}")


def test_remove_padding():
    X, y = get_files('C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project')
    frames, action_segment_td, ground_truth_actions = read_data(X, y)
    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)
    frames = standardise_features(frames)

    lens = [len(f) for f in frames]
    max_len = max([len(f) for f in frames])
    print(max_len)
    actions_per_seq, label_to_index_map = encode_labels(frames)
    # print(label_to_index_map)
    padded_numeric_features, padded_labels = pad_data(frames, actions_per_seq)
    random_index = random.randint(0, 30)
    test_X, test_y = torch.unsqueeze(padded_numeric_features[random_index], 0), torch.unsqueeze(
        padded_labels[random_index], 0)
    # print(test_X.shape)
    # print(test_y.shape)

    # checking if tensors are un-padded to original lengths
    print(lens)
    outputs, targets = remove_padding(test_X, test_y)

    # print(outputs.shape)
    # print(targets.shape)
    assert outputs.shape[0] in lens
    assert targets.shape[0] in lens


def test_preprocess_dataset(PATH_TO_DIR = 'C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project' ):
  files, labels = get_files(PATH_TO_DIR)

  frames, action_segment_td, ground_truth_actions = read_data(files, labels)
  print(len(frames))
  print(len(action_segment_td))
  print(len(ground_truth_actions))
  frames = standardise_features(append_labels_per_frame(frames, action_segment_td, ground_truth_actions))
  print(len(frames))
  actions_per_seq, index_label_map = encode_labels(frames)
  print(len(actions_per_seq))

  #X_data, y_data = pad_data(frames,actions_per_seq)

  # return X_data, y_data