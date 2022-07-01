import glob
import pandas as pd
import numpy as np

def get_files(PATH_TO_DIR):
    files = glob.glob(f"{PATH_TO_DIR}/dataset/avocado/clutter/[0-9]*/optoforce_data.csv")
    labels = glob.glob(f"{PATH_TO_DIR}/dataset/avocado/clutter/[0-9]*/labels")
    return files, labels


def read_data(files, labels):
    """reads every 830th frame which is approx 1FPS"""
    frames = []
    for file in files:
        data_df = pd.read_csv(file)


        data_df = data_df.iloc[::830, :]
        #data_df['index'] = data_df[['index_x','index_y','index_z']].apply(lambda x: np.sqrt(x.dot(x)), axis = 1)
        data_df['index'] = np.linalg.norm(data_df[['index_x','index_y','index_z']].values,axis=1)
        data_df['middle'] = np.linalg.norm(data_df[['midle_x','middle_y','middle_z']].values,axis=1)
        data_df['thumb'] = np.linalg.norm(data_df[['thumb_x','thumb_y','thumb_z']].values,axis=1)

        data_df = data_df.drop(columns=['ring_x', 'ring_y', 'ring_z',
                                        'index_x', 'index_y', 'index_z',
                                        'middle_x', 'middle_y', 'middle_z',
                                        'thumb_x', 'thumb_y', 'thumb_z'])
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

def main():
    PATH_TO_DIR = 'C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project'
    files,lables = get_files(PATH_TO_DIR)

    frames, _, _ = read_data(files,lables)

    print(frames[0].iloc[60])

if __name__ == '__main__':
    main()

