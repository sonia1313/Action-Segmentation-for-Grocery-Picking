import random

import torch

from utils.preprocessing import *
import numpy as np
import pytest
import pandas as pd
import datatest as dt


def test_get_files():
    X, y = get_files('C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project')
    #
    # for i,j in zip(X,y):
    #     print(i)
    #     print(j)
    # #print(y)

    print(X[0])
    assert len(X) == 30
    assert len(y) == 30


def test_read_data():
    X, y = get_files('C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project')
    frames, action_segment_td, ground_truth_actions = read_data(X, y)

    assert len(frames[0].columns) == 5
    # assert np.all(frames[0].columns == ['time', 'index', 'middle','thumb','label'])
    dt.validate(frames[0].columns, {'time', 'index', 'middle', 'thumb', 'label'})
    assert len(action_segment_td) == len(ground_truth_actions)
    dt.validate(frames[0].label, str)


def test_encode_labels():
    X, y = get_files('C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project')
    frames, action_segment_td, ground_truth_actions = read_data(X, y)
    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)
    frames = standardise_features(frames)

    actions_per_seq, unique_actions, label_to_index_map = encode_labels(frames)
    print(actions_per_seq[0])
    assert len(unique_actions) == 6
    dt.validate(actions_per_seq[0], int)


def test_pad_data():
    X, y = get_files('C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project')
    frames, action_segment_td, ground_truth_actions = read_data(X, y)
    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)
    frames = standardise_features(frames)

    max_len = max([len(f) for f in frames])
    print(max_len)
    actions_per_seq, unique_actions, label_to_index_map = encode_labels(frames)
    # print(label_to_index_map)
    padded_numeric_features, padded_labels = pad_data(frames, actions_per_seq)

    assert len(padded_numeric_features[0]) == len(padded_labels[0])
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
    actions_per_seq, unique_actions, label_to_index_map = encode_labels(frames)
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
  actions_per_seq, unique_actions, index_label_map = encode_labels(frames)
  print(len(actions_per_seq))

  #X_data, y_data = pad_data(frames,actions_per_seq)

  # return X_data, y_data