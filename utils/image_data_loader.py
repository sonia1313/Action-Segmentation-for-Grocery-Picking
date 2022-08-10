import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import VisionDataset

from utils.tactile_preprocessing import *


class ImageDataset(Dataset):

    def __init__(self, frames_per_seq, transform,size):
        """
        :param sequences: [[img_frames_seq_1][img_frames_seq_2].....[imag_frames_seq_n]]
        :param actions: [[actions_per_frame_seq_1]..[actions_per_frame_seq_2]....[actions_per_frame_seq_n]]
        """
        self.frames_per_seq = frames_per_seq
        self.label_to_index_map = {'move-in': 0, 'manipulate': 1, 'grasp': 2, 'pick-up': 3, 'move-out': 4, 'drop': 5}
        self.transform = transform
        self.size = size

    def __getitem__(self, seq_idx):
        seq_df = self.frames_per_seq[seq_idx]

        images_in_seq = torch.zeros(size=(len(seq_df), 3, self.size, self.size))  # nchannel, h, w
        actions_in_seq = torch.zeros(size=(len(seq_df),))
        # env_in_seq = []
        # fruit_in_seq = []

        for i in range(len(seq_df)):
            img_pth = seq_df['path'].iloc[i]

            pil_img = Image.open(img_pth)
            # img_ = read_image(path=img_pth) # returns a tensor of shape image_channels, image_height, image_width

            images_in_seq[i] = self.transform(pil_img)

            label = seq_df['label'].iloc[i]

            actions_in_seq[i] = self.label_to_index_map[label]

        return images_in_seq, actions_in_seq

    def __len__(self):
        return len(self.frames_per_seq)


def load_data(files, single, clutter, size=32, seed=42, batch_size=1):
    # train_size = int(0.6 * len(x_data))
    # val_size, test_size = int(0.2 * len(x_data)), int(0.2 * len(x_data))
    if single is True and clutter is False:
        train_size = 20
        val_size = 5
        test_size = 4



    elif single is False and clutter is True:
        train_size = 20
        val_size = 5
        test_size = 5


    else:
        train_size = 50
        val_size = 5
        test_size = 4

    transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.48, 0.5, 0.48], [0.335, 0.335, 0.335])])

    dataset = ImageDataset(files, transform=transform, size=size)
    # print(dataset[0])
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
