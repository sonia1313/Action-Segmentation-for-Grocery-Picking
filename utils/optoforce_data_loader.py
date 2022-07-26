from torch.utils.data import Dataset, DataLoader, random_split, Subset
from utils.preprocessing import *


class OpToForceDataset(Dataset):
    def __init__(self, sequences, actions):
        self.X = sequences
        self.y = actions

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]


def load_data(x_data, y_data, train_size = 5, val_size = 1,test_size = 1):
    #train_size = int(0.6 * len(x_data))
    #val_size, test_size = int(0.2 * len(x_data)), int(0.2 * len(x_data))

    dataset = OpToForceDataset(x_data, y_data)
    #print(dataset[0])
    #train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataset = Subset(dataset, [0,1,2,3,4])
    val_dataset = Subset(dataset, [5])
    test_dataset = Subset(dataset, [6])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
