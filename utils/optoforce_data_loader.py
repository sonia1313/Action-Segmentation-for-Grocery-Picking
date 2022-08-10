import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets import VisionDataset

from utils.tactile_preprocessing import *
#torch.manual_seed(42)

class OpToForceDataset(Dataset):
    def __init__(self, sequences, actions):
        self.X = sequences
        self.y = actions

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]


def load_data(x_data, y_data, single, clutter, seed, batch_size=1):
    # train_size = int(0.6 * len(x_data))
    # val_size, test_size = int(0.2 * len(x_data)), int(0.2 * len(x_data))
    if single == True and clutter == False:
        train_size = 20
        val_size = 5
        test_size = 4

    elif single == False and clutter == True:
        train_size = 20
        val_size = 5
        test_size = 5
    else:
        train_size = 50
        val_size = 5
        test_size = 4

    dataset = OpToForceDataset(x_data, y_data)
    # print(dataset[0])
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(seed))
    #train_dataset = Subset(dataset, [0,1,2,3,4])
    #val_dataset = Subset(dataset, [5,8])
    #test_dataset = Subset(dataset, [6,7])


    # train_dataset = Subset(dataset, [0,35,2,33,4,42])
    # val_dataset = Subset(dataset, [5,45])
    # test_dataset = Subset(dataset, [6,48])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


