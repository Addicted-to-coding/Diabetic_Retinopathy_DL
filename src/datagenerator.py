from torch.utils.data import Dataset
from augmentation_utils import *
import numpy as np
import pandas as pd
from PIL import Image
from os import path

random.seed(10)

def get_dataset(path, dir, train_size = 0.8, valid_size = 0.1):

    data = pd.read_csv(path)
    arr = [i for i in range(len(data))]
    random.shuffle(arr)
    train_indices = arr[: int(train_size * len(data)) ]
    train_X, train_y = get_image_label(data, dir, train_indices)
    train_X = train_X / 255.0

    valid_indices = arr[int(train_size * len(data)) : int((train_size + valid_size) * len(data))]
    valid_X, valid_y = get_image_label(data, dir, valid_indices)
    valid_X = valid_X / 255.0

    test_indices = arr[int((train_size + valid_size) * len(data)) : ]
    test_X, test_y = get_image_label(data, dir, test_indices)
    test_X = test_X / 255.0

    return train_X, train_y, valid_X, valid_y, test_X, test_y

def get_image_label(data, dir, indices):

    X = np.array([np.asarray(Image.open(f'{dir}{data.iloc[i].id_code}.png')) for i in indices])
    y = np.array([int(data.iloc[i].diagnosis) for i in indices])
    return X,y

class DataGenerator(Dataset):
    def __init__(self, X, y, mixup = False, cutmix = False):
        super(DataGenerator, self).__init__()
        self.X = np.transpose(X, (0, 3, 1, 2))
        self.y = y

        if mixup:
        	self.X, self.y = add_mixup_data(self.X, self.y)
        elif cutmix:
        	self.X, self.y = add_cutmix_data(self.X, self.y)

        self.length = len(X)

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):
        return self.length
