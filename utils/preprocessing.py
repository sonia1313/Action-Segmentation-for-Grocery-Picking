import glob
import os
import re

import pandas as pd
import numpy as np

#
# #if working in colab
# os.chdir("/content/drive/Othercomputers/Dell")
#
# PATH_TO_DIR = os.chdir("/content/drive/Othercomputers/Dell")
import torch


def get_files(path):
    # files = glob.glob(f"{PATH_TO_DIR}/dataset/*/clutter/[0-9]*/optoforce_data.csv")
    # labels = glob.glob(f"{PATH_TO_DIR}/dataset/*/clutter/[0-9]*/labels")
    files = glob.glob(f"{path}/*/clutter/[0-9]*/optoforce_data.csv")
    labels = glob.glob(f"{path}/*/clutter/[0-9]*/labels")
    return files, labels


def read_data(files, labels, fps=1):
    """ 840 frames is approx 1FPS"""
    #840
    nth_frame = 840 // fps
    frames = []
    for file in files:
        fruit_and_env = re.findall('avocado|banana|blueberry|clutter|single', file)
        data_df = pd.read_csv(file)
        # data_df = data_df.drop(columns=['ring_x', 'ring_y', 'ring_z'])

        data_df = data_df.iloc[::nth_frame, :]  #
        data_df['index'] = np.linalg.norm(data_df[['index_x', 'index_y', 'index_z']].values, axis=1)
        data_df['middle'] = np.linalg.norm(data_df[['middle_x', 'middle_y', 'middle_z']].values, axis=1)
        data_df['thumb'] = np.linalg.norm(data_df[['thumb_x', 'thumb_y', 'thumb_z']].values, axis=1)

        data_df = data_df.drop(columns=['ring_x', 'ring_y', 'ring_z',
                                        'index_x', 'index_y', 'index_z',
                                        'middle_x', 'middle_y', 'middle_z',
                                        'thumb_x', 'thumb_y', 'thumb_z'])
        data_df["fruit"] = fruit_and_env[0]
        data_df["environment"] = fruit_and_env[1]
        data_df["label"] = ""
        frames.append(data_df)

    action_segment_td = []  # time durations for each action
    ground_truth_actions = []  # action per frame

    for labels_per_file in labels:
        td_per_file = []
        gt_actions_per_file = []
        with open(labels_per_file) as f:
            for line in f:
                x, y = line.split(';')
                td_per_file.append(x)
                gt_actions_per_file.append(y.strip('\n'))
            action_segment_td.append(td_per_file)
            ground_truth_actions.append(gt_actions_per_file)

    return frames, action_segment_td, ground_truth_actions


def append_labels_per_frame(frames, action_segment_td, ground_truth_actions):
    for df, duration_of_actions, labels in zip(frames, action_segment_td, ground_truth_actions):
        condition = []
        for actions in duration_of_actions:
            start_time, end_time = actions.split(':')
            condition.append(df['time'].between(int(start_time), int(end_time)))

        df['label'] = np.select(condition, labels, default=None)
        df.dropna(inplace=True)

    return frames


def standardise_features(frames,
                         features=['index', 'middle', 'thumb']):
    for frame in frames:
        for feature in frame[features]:
            mean = frame[feature].mean()
            std = frame[feature].std()

            frame[feature] = (frame[feature] - mean) / std

    return frames


def encode_labels(frames):
    # unique_actions = set()

    label_to_index_map = {'move-in': 0, 'manipulate': 1, 'grasp': 2, 'pick-up': 3, 'move-out': 4, 'drop': 5}
    # for frame in frames:
    #     for label in frame['label']:
    #         unique_actions.add(label)
    # print(unique_actions)
    #one_hot_encoding_acts = pd.get_dummies(list(unique_actions))
    #label_to_index_map = {k: np.argmax(v) for k, v in one_hot_encoding_acts.items()}
    # print(label_to_index_map)
    actions_per_seq = []
    for frame in frames:
        action_encodings = []
        for i in range(0, len(frame)):
            # action_encodings.append(one_hot_encoding_acts[frame['label'].iloc[i]])
            action_encodings.append(label_to_index_map[frame['label'].iloc[i]])
        actions_per_seq.append(action_encodings)

    return actions_per_seq, label_to_index_map


def pad_data(frames, actions_per_seq, n_features=3, n_classes=6, n_sequences=30):
    features = ['index', 'middle', 'thumb']
    max_length = max([len(frame) for frame in frames])
    numeric_features_per_seq = [np.array(frames[i][features]) for i in range(len(frames))]

    labels_per_seq = [np.array(actions_per_seq[i]) for i in range(len(actions_per_seq))]

    padded_numeric_features_per_seq = np.zeros((n_sequences, max_length, n_features))
    padded_labels_per_seq = -1 * np.ones((n_sequences, max_length,))
    for i in range(len(numeric_features_per_seq)):
        last_timestep = numeric_features_per_seq[i][-1:][0]
        repeat_n = max_length - numeric_features_per_seq[i].shape[0]
        padding = np.tile(last_timestep, (repeat_n, 1))
        # print(padding)
        padded_seq = np.concatenate((numeric_features_per_seq[i], padding), axis=0)
        padded_numeric_features_per_seq[i] = padded_seq

    for i in range(len(labels_per_seq)):
        seq_len = labels_per_seq[i].shape[0]

        padded_labels_per_seq[i][:seq_len] = labels_per_seq[i]

    return torch.FloatTensor(padded_numeric_features_per_seq), torch.LongTensor(padded_labels_per_seq)


def remove_padding(predictions_padded, targets_padded):
    print(f"shape of padded targets {targets_padded.shape}")
    mask = (targets_padded >= 0).long()  # only outputs labels that is >= 0
    # print(mask)
    print(f"shape of mask {mask.shape}")

    n = len([out for out in mask.squeeze() if out.all() >= 1])
    # print(n)
    outputs = predictions_padded.squeeze()[:n, :]
    # print(f"unpadded outputs {outputs.shape}")

    targets_padded = targets_padded.squeeze()
    targets = targets_padded[:n]
    # print(f"unpadded targets {targets.shape}")
    # _, targets = targets.max(dim=1)  # remove one hot encoding

    return outputs.unsqueeze(0), targets.unsqueeze(0)


def preprocess_dataset(cfg_preprocess):

    files, labels = get_files(cfg_preprocess['data_path'])

    frames, action_segment_td, ground_truth_actions = read_data(files, labels, cfg_preprocess['frames_per_sec'])
    frames = standardise_features(append_labels_per_frame(frames, action_segment_td, ground_truth_actions))
    actions_per_seq, label_to_index_map = encode_labels(frames)
    X_data, y_data = pad_data(frames, actions_per_seq, n_features=cfg_preprocess['n_features'],
                              n_sequences=cfg_preprocess['n_sequences'],
                              )

    return X_data, y_data, label_to_index_map
