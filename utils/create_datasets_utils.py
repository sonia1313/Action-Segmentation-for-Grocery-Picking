""""
Author - Sonia Mathews
create_datasets_utils.py

Script to create pickle files of datasets at different fps,
to improve efficiency during experimentation
"""
import os

import torch

os.chdir("C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project")

from utils.preprocessing import preprocess_dataset

preprocess_config = {
    'data_path': 'dataset',  # get_files()
    'clutter': True,  # get_files()
    'single': True,  # get_files()
    'frames_per_sec': 15, #6 #3 #1 #15  #read_data()
    'feature_engineering': True,  # True #read_data()
    'standardise_data': False,  # preprocess_dataset()
    'normalize_data': True,  # preprocess_dataset()
    'pad_data': True  # preprocess_dataset()
}


def get_dataset():
    X_data, y_data, _ = preprocess_dataset(preprocess_config)

    print(type(X_data))
    print(type(y_data))
    print(f"no_sequences:{len(X_data)}")

    torch.save((X_data,y_data), 'data/tactile/with_feature_engineering/single_clutter_15fps.pt')
    X, y = torch.load('data/tactile/with_feature_engineering/single_clutter_15fps.pt')
    print(X.shape)
    print(y.shape)


if __name__ == '__main__':
    get_dataset()
