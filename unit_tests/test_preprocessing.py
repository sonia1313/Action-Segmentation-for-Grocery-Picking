import os
import random
import re

import pytest
import torch
import yaml

from models.encoder_decoder_lstm import EncoderDecoderLSTM, LitEncoderDecoderLSTM
from utils.preprocessing import *
import numpy as np
import datatest as dt

CONFIG_PATH = "config"

os.chdir("C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project")


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as f:
        config = yaml.safe_load(f)

        return config


@pytest.fixture
def config_preprocess():
    config = load_config('encoder_decoder_baseline.yaml')
    # print(config['dataset']['preprocess'])

    return config['dataset']['preprocess']


def test_get_files(config_preprocess):
    # checking congif file and clutter dataset
    X, y = get_files(path=config_preprocess['data_path'], clutter=config_preprocess['clutter'],
                     single=config_preprocess['single'])

    assert len(X) == 30
    assert len(y) == 30
    assert type(X[0]) == str
    assert type(y[0]) == str

    # cheking single dataset
    X, y = get_files(path=config_preprocess['data_path'], clutter=False,
                     single=True)
    assert len(X) == 29
    assert len(y) == 29

    X, y = get_files(path=config_preprocess['data_path'], clutter=False,
                     single=False)
    assert len(X) == 59
    assert len(y) == 59


def test_read_data(config_preprocess):
    # clutter = False
    # single = True
    X, y = get_files(path=config_preprocess['data_path'], clutter=config_preprocess['clutter'],
                     single=config_preprocess['single'])

    # testing config values
    frames, action_segment_td, ground_truth_actions, features = read_data(files=X, labels=y, fps=1,
                                                                          feature_engineering=config_preprocess[
                                                                              'feature_engineering'])

    # assert len(frames[0].columns) == 7
    # # assert np.all(frames[0].columns == ['time', 'index', 'middle','thumb','label'])
    dt.validate(features, {'index', 'middle', 'thumb'})
    print(features)
    # assert len(action_segment_td) == len(ground_truth_actions)
    dt.validate(frames[0].label, str)
    dt.validate(frames[0].fruit, str)
    dt.validate(frames[0].environment, str)
    print()
    print()
    print(frames[0].head())

    # testing when feature engineering is False
    frames, action_segment_td, ground_truth_actions, features = read_data(files=X, labels=y, fps=1,
                                                                          feature_engineering=False)

    dt.validate(features, {'index_x', 'index_y', 'index_z',
                           'middle_x', 'middle_y', 'middle_z',
                           'thumb_x', 'thumb_y', 'thumb_z'})
    print(features)


def test_append_labels_per_frame(config_preprocess):
    X, y = get_files(path=config_preprocess['data_path'], clutter=config_preprocess['clutter'],
                     single=config_preprocess['single'])

    frames, action_segment_td, ground_truth_actions, _ = read_data(files=X, labels=y, fps=1,
                                                                   feature_engineering=config_preprocess[
                                                                       'feature_engineering'])
    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)
    print(frames[0].head().label)
    print(frames[0].tail().label)

    assert set(frames[0].label) == {'move-in', 'manipulate', 'grasp', 'pick-up', 'move-out', 'drop'}
    random_int = random.randint(0, len(X))
    assert set(frames[random_int].label) == {'move-in', 'manipulate', 'grasp', 'pick-up', 'move-out', 'drop'}


def test_standardise_features(config_preprocess):
    X, y = get_files(path=config_preprocess['data_path'], clutter=config_preprocess['clutter'],
                     single=config_preprocess['single'])
    frames, action_segment_td, ground_truth_actions, features = read_data(files=X, labels=y, fps=1,
                                                                          feature_engineering=config_preprocess[
                                                                              'feature_engineering'])

    # print(features)
    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)
    # frames = standardise_features(frames, features)

    # standardisation
    for frame in frames:
        for feature in frame[features]:
            mean = frame[feature].mean()
            std = frame[feature].std()

            frame[feature] = (frame[feature] - mean) / std  # 1

    print(frames[random.randint(0, len(X))])


# min_max_scaling
def test_normalise_features(config_preprocess):
    X, y = get_files(path=config_preprocess['data_path'], clutter=config_preprocess['clutter'],
                     single=config_preprocess['single'])
    frames, action_segment_td, ground_truth_actions, features = read_data(files=X, labels=y, fps=1,
                                                                          feature_engineering=config_preprocess[
                                                                              'feature_engineering'])
    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)

    for frame in frames:
        for feature in frame[features]:
            min_val = frame[feature].min()
            max_val = frame[feature].max()

            frame[feature] = (frame[feature]-min_val)/(max_val-min_val)

    print(frames[random.randint(0, len(X))][features])

def test_encode_labels():
    X, y = get_files(path=config_preprocess['data_path'], clutter=config_preprocess['clutter'],
                     single=config_preprocess['single'])
    frames, action_segment_td, ground_truth_actions, features = read_data(files=X, labels=y, fps=1,
                                                                          feature_engineering=config_preprocess[
                                                                              'feature_engineering'])

    # print(features)
    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)
    frames = standardise_features(frames, features)

    actions_per_seq, label_to_index_map = encode_labels(frames)

    dt.validate(actions_per_seq[0], int)
    random_int = random.randint(0, 10)
    for i in range(0, len(X)):
        print(i)
        assert len(frames[i] == len(actions_per_seq[i]))
    indices = set(label_to_index_map)
    print(indices)
    for i in range(0, len(X)):
        for encoded_label, label in zip(actions_per_seq[i], frames[i].label):
            assert label_to_index_map[label] == encoded_label

    for encoded_label, label in zip(actions_per_seq[random_int], frames[random_int].label):
        print(f"{label} and {encoded_label}")


def test_pad_data(config_preprocess):
    X, y = get_files(path=config_preprocess['data_path'], clutter=config_preprocess['clutter'],
                     single=config_preprocess['single'])
    frames, action_segment_td, ground_truth_actions, features = read_data(files=X, labels=y, fps=1,
                                                                          feature_engineering=config_preprocess[
                                                                              'feature_engineering'])

    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)
    frames = standardise_features(frames, features)

    max_len = max([len(f) for f in frames])
    print(max_len)

    actions_per_seq, label_to_index_map = encode_labels(frames)
    padded_numeric_features, padded_labels = pad_data(frames, actions_per_seq, n_sequences=len(X), features=features)

    assert len(padded_numeric_features[0]) == len(padded_labels[0])
    print()
    print(label_to_index_map)

    print(padded_numeric_features)
    print(padded_labels[0])

    random_int = random.randint(0, len(X))
    assert (len(padded_labels[random_int]) == len(padded_labels[random_int]))

    print(f"max of sequence length {len(padded_labels[0])}")


def test_remove_padding(config_preprocess):
    X, y = get_files(path=config_preprocess['data_path'], clutter=config_preprocess['clutter'],
                     single=config_preprocess['single'])
    frames, action_segment_td, ground_truth_actions, features = read_data(files=X, labels=y, fps=1,
                                                                          feature_engineering=config_preprocess[
                                                                              'feature_engineering'])
    frames = append_labels_per_frame(frames, action_segment_td, ground_truth_actions)
    frames = standardise_features(frames, features)

    lens = [len(f) for f in frames]
    max_len = max([len(f) for f in frames])
    print(max_len)
    actions_per_seq, label_to_index_map = encode_labels(frames)
    # print(label_to_index_map)
    padded_numeric_features, padded_labels = pad_data(frames, actions_per_seq, features, len(X))

    sample_X, sample_y = torch.unsqueeze(padded_numeric_features[0], dim=0), torch.unsqueeze(padded_labels[0], dim=0)

    model = LitEncoderDecoderLSTM(n_features=3, hidden_size=100, n_layers=1, experiment_tracking=False)

    logits = model(sample_X, sample_y, teacher_forcing=0.0)

    # print(logits.shape) #logits shape [1,seq_len,n_classes]

    random_index = random.randint(0, 30)
    test_X, test_y = torch.squeeze(padded_numeric_features[random_index], 0), torch.squeeze(
        padded_labels[random_index], 0)
    # print(test_X.shape)
    # print(test_y.shape)

    # checking if tensors are un-padded to original lengths
    print(lens)
    outputs, targets = remove_padding(test_X, test_y)

    print(outputs.shape)
    print(targets.shape)
    assert outputs.shape[0] in lens
    assert targets.shape[0] in lens

# def test_preprocess_dataset(
#         PATH_TO_DIR='C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project'):
#     files, labels = get_files(PATH_TO_DIR)
#
#     frames, action_segment_td, ground_truth_actions = read_data(files, labels)
#     print(len(frames))
#     print(len(action_segment_td))
#     print(len(ground_truth_actions))
#     frames = standardise_features(append_labels_per_frame(frames, action_segment_td, ground_truth_actions))
#     print(len(frames))
#     actions_per_seq, index_label_map = encode_labels(frames)
#     print(len(actions_per_seq))
#
#     # X_data, y_data = pad_data(frames,actions_per_seq)
#
#     # return X_data, y_data
