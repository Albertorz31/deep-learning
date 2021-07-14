from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.mode = str(mode)
        self.data = data 
        self._transform = tv.transforms.Compose(
            [tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data.values[index][0]
        image = imread(img_name)
        label = torch.tensor([self.data.values[index][1], self.data.values[index][2]])
        image_rgb = gray2rgb(image)
        image_rgb_transformed = self._transform(image_rgb)

        sample = (image_rgb_transformed, label)
        return sample
