from cProfile import label

# from cv2 import transform
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import random

from PIL import Image, ImageOps

# import any other libraries you need below this line
from torchvision import transforms
from math import floor


class RandomGammaCorrection:
    def __call__(self, image):
        gammas = [0, 0, 0, 0.5, 1, 1.5, 0.75, 1.25]
        gamma = random.choice(gammas)

        if gamma == 0:
            return image

        else:
            return transforms.functional.adjust_gamma(image, gamma, gain=1)


class Cell_data(Dataset):
    def __init__(self, data_dir, size, train=True, train_test_split=0.8, shuffle=False):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        # todo
        # initialize the data class

        self.train = train

        label_dir = os.path.join(data_dir, "labels")
        scans_dir = os.path.join(data_dir, "scans")
        self.images = [os.path.join(scans_dir, f) for f in os.listdir(scans_dir)]
        self.labels = [os.path.join(label_dir, f) for f in os.listdir(label_dir)]

        if shuffle:
            random.shuffle(self.images)
            random.shuffle(self.labels)

        N = floor(len(self.images) * train_test_split)
        if train:
            self.images = self.images[:N]
            self.labels = self.labels[:N]
        else:
            self.images = self.images[N + 1 :]
            self.labels = self.labels[N + 1 :]

        self.to_tensor = transforms.ToTensor()
        self.transform = transforms.Compose(  # compose
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                RandomGammaCorrection(),
                transforms.RandomRotation(15),
                # transforms.RandomResizedCrop(size, (0.5, 1.5)),
            ]
        )

        self.resize = transforms.Resize(size)

    def __getitem__(self, idx):
        # todo
        # load image and mask from index idx of your data
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])
        image = self.to_tensor(image)
        label = self.to_tensor(label)

        image = self.resize(image)
        label = self.resize(label)

        if self.train:
            image = self.transform(image)
            label = self.transform(label)

        label[label != 0] = 1
        return image, label

    def __len__(self):
        return len(self.images)
