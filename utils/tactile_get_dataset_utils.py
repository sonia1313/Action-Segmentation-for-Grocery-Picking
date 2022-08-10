""""
Author - Sonia Mathews
tactile_get_dataset_utils.py

Script to create datasets to pass to OpToForceDataset() class
to improve efficiency during experimentation
"""
import os

import torch
import pickle


from utils.tactile_preprocessing import encode_labels, pad_data, standardise_features, normalize_features

# preprocess_config = {
#     'data_path': 'dataset',  # get_files()
#     'clutter': True,  # get_files()
#     'single': True,  # get_files()
#     'frames_per_sec': 15, #6 #3 #1 #15  #read_data()
#     'feature_engineering': True,  # True #read_data()
#     'standardise_data': False,  # preprocess_dataset()
#     'normalize_data': True,  # preprocess_dataset()
#     'pad_data': True}  # preprocess_dataset()

def get_dataset():
    os.chdir("C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project")
    print(os.getcwd())
    #3fps_clutter_single

    with open('data/tactile/with_feature_engineering/3fps/3fps_clutter_tactile_df.pkl', 'rb') as f:
        clutter_dfs = pickle.load(f)
        # print(len(frames))
        print(clutter_dfs[0].head())
        actions_per_seq, label_to_index_map = encode_labels(clutter_dfs)

        X_data, y_data = pad_data(clutter_dfs, actions_per_seq, n_sequences= len(clutter_dfs), features=['index','middle','thumb'])

    print(type(X_data))
    print(type(y_data))
    print(f"no_sequences:{len(X_data)}")

    torch.save((X_data,y_data), 'data/tactile/with_feature_engineering/3fps/clutter_3fps_dataset.pt')
    X, y = torch.load('data/tactile/with_feature_engineering/3fps/clutter_3fps_dataset.pt')
    print(X.shape)
    print(y.shape)
    print(X[0])
    print(y[0])
    print(len(X[0]))
    print(len(y[0]))


if __name__ == '__main__':
    get_dataset()
