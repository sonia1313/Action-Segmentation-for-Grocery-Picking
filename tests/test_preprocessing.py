import random

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
    assert  len(y) == 30

def test_read_data():
    X, y = get_files('C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project')
    frames, action_segment_td, ground_truth_actions = read_data(X,y)

    assert len(frames[0].columns) == 5
    #assert np.all(frames[0].columns == ['time', 'index', 'middle','thumb','label'])
    dt.validate(frames[0].columns, {'time', 'index', 'middle','thumb','label'} )
    assert len(action_segment_td) == len(ground_truth_actions)
    dt.validate(frames[0].label,str)
def test_one_hot_encode_labels():
    X, y = get_files('C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project')
    frames,action_segment_td, ground_truth_actions = read_data(X, y)
    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)
    frames = standardise_features(frames)

    actions_per_seq, unique_actions, label_to_index_map = one_hot_encode_labels(frames)
    print(actions_per_seq[0])
    assert len(unique_actions) == 6
    dt.validate(actions_per_seq[0],int)

def test_pad_data():
    X, y = get_files('C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project')
    frames,action_segment_td, ground_truth_actions = read_data(X, y)
    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)
    frames = standardise_features(frames)

    actions_per_seq, unique_actions, label_to_index_map = one_hot_encode_labels(frames)
    print(label_to_index_map)
    padded_numeric_features, padded_labels = pad_data(frames,actions_per_seq)

    assert len(padded_numeric_features[0]) == len(padded_labels[0])
    print(padded_labels[0])

    assert (len(padded_labels[np.random.randint(0,30)]) == len(padded_labels[np.random.randint(0,30)]))

    print(f"max of sequence length {len(padded_labels[0])}")