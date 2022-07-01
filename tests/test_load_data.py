import pytest
from utils.optoforce_data_loader import load_data, OpToForceDataset
from utils.preprocessing import *
from torch.utils.data import random_split, DataLoader

PATH_TO_DIR = 'C:/Users/sonia/OneDrive - Queen Mary, University of London/Action-Segmentation-Project'


def test_op_to_force_dataset(train_size = 20, val_size = 5, test_size = 5):
    X_data, y_data = preprocess_dataset(PATH_TO_DIR)

    dataset = OpToForceDataset(X_data, y_data)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    test_X, test_y = dataset[0]

    print(test_X.shape)
    print(test_y.shape)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # for data, target in train_loader:
    #     print(data.shape)
    #     print(target.shape)
    #     break

    #testing implementaion frol pl

    for batch_idx, batch in enumerate(train_loader):
        print(batch_idx)
        #print(batch)
        test_X, test_y = batch

        print(test_X.shape)
        print(test_y.shape)

        break



def test_load_data():
    X_data, y_data = preprocess_dataset(PATH_TO_DIR)
    train_loader, val_loader, test_loader = load_data(X_data, y_data)

    test_X, test_y = print(next(train_loader))
    print(test_X.shape)
    print(test_y.shape)

