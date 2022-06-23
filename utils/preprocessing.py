import glob
import os
import pandas as pd
import numpy as np

#
# #if working in colab
# os.chdir("/content/drive/Othercomputers/Dell")
#
# PATH_TO_DIR = os.chdir("/content/drive/Othercomputers/Dell")
import torch


def get_files(PATH_TO_DIR):
    files = glob.glob(f"{PATH_TO_DIR}/dataset/avocado/clutter/[0-9]*/optoforce_data.csv")
    labels = glob.glob(f"{PATH_TO_DIR}/dataset/avocado/clutter/[0-9]*/labels")
    return files, labels


def read_data(files, labels):
    """reads every 830th frame which is approx 1FPS"""
    frames = []
    for file in files:
        data_df = pd.read_csv(file)
        data_df = data_df.drop(columns=['ring_x', 'ring_y', 'ring_z'])
        data_df = data_df.iloc[::830, :]
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
                         features=['index_x', 'index_y', 'index_z', 'middle_x', 'middle_y', 'middle_z', 'thumb_x',
                                   'thumb_y', 'thumb_z']):
    for frame in frames:
        for feature in frame[features]:
            mean = frame[feature].mean()
            std = frame[feature].std()

            frame[feature] = (frame[feature] - mean) / std

    return frames


def one_hot_encode_labels(frames):
    unique_actions = set()

    for frame in frames:
        for label in frame['label']:
            unique_actions.add(label)

    one_hot_encoding_acts = pd.get_dummies(list(unique_actions))
    index_label_map = {np.argmax(v): k for k, v in one_hot_encoding_acts.items()}

    actions_per_seq = []
    for frame in frames:
        action_encodings = []
        for i in range(0, len(frame)):
            action_encodings.append(one_hot_encoding_acts[frame['label'].iloc[i]])
        actions_per_seq.append(action_encodings)

    return actions_per_seq, unique_actions, index_label_map


def pad_data(frames, actions_per_seq):
    features = ['index_x', 'index_y', 'index_z', 'middle_x', 'middle_y', 'middle_z', 'thumb_x', 'thumb_y', 'thumb_z']
    max_length = max([len(frame) for frame in frames])
    numeric_features_per_seq = [np.array(frames[i][features]) for i in range(len(frames))]

    labels_per_seq = [np.array(actions_per_seq[i]) for i in range(len(actions_per_seq))]

    padded_numeric_features_per_seq = np.zeros((10, max_length, 9))
    padded_labels_per_seq = -1 * np.ones((10, max_length, 6))
    for i in range(len(numeric_features_per_seq)):
        last_timestep = numeric_features_per_seq[i][-1:][0]
        repeat_n = max_length - numeric_features_per_seq[i].shape[0]
        padding = np.tile(last_timestep, (repeat_n, 1))
        # print(padding)
        padded_seq = np.concatenate((numeric_features_per_seq[i], padding), axis=0)
        padded_numeric_features_per_seq[i] = padded_seq

    for i in range(len(labels_per_seq)):
        seq_len = labels_per_seq[i].shape[0]

        padded_labels_per_seq[i][:seq_len, :] = labels_per_seq[i]

    return torch.FloatTensor(padded_numeric_features_per_seq), torch.LongTensor(padded_labels_per_seq)
