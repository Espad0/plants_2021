import pandas as pd
import numpy as np
import torch
import os
from skimage import io, transform
import cv2
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

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

        img = cv2.resize(image, (new_h, new_w))/255.

        return {'image': img, 'label': label}

def build_sampler(csv_path):
    train_df = pd.read_csv(csv_path)
    target = train_df.labels.values

    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)]
    )

    class_weigths = {}
    for i, unique_label in enumerate(np.unique(target)):
        class_weigths[unique_label] = 1 / class_sample_count[i]

    sample_weigths = np.array([class_weigths[t] for t in target])
    sampler_weigths = torch.Tensor(sample_weigths)
    sampler = WeightedRandomSampler(sampler_weigths, len(sampler_weigths))
    return sampler