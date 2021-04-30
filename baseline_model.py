import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from skimage import io, transform


input_size = 256
number_of_maxpools = 3
fc_size = int(input_size / (2**number_of_maxpools))


labels_dict = {'healthy':0,
               'scab frog_eye_leaf_spot complex':1,
               'scab':2,
               'complex':3,
               'rust':4,
               'frog_eye_leaf_spot':5,
               'powdery_mildew':6,
               'scab frog_eye_leaf_spot':7,
               'frog_eye_leaf_spot complex':8,
               'rust frog_eye_leaf_spot':9,
               'powdery_mildew complex':10,
               'rust complex':11}

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*fc_size*fc_size, 12)
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.plant_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.plant_labels.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                self.plant_labels.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.plant_labels.iloc[idx, 1]

        sample = {'image': image, 'label': labels_dict[label]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': label}

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label}