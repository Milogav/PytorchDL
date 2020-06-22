import os
import glob
import random

import cv2
import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, output_shape):
        """Data directory must contain one image directory per class, name after the class name.

        :param data_dir: directory containing the image and label data.
        :param output_shape: list or tuple defining the generator output shape
        """

        self.data_files = []
        self.class_tags = sorted(os.listdir(data_dir))

        for i, tag in enumerate(self.class_tags):
            class_dir = os.path.join(data_dir, tag)
            img_files = glob.glob(os.path.join(class_dir, '*'))
            for path in img_files:
                self.data_files.append([path, i])

        self.shuffle()
        self.output_shape = output_shape

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        h, w, ch = self.output_shape

        img_path, label = self.data_files[index]
        if ch > 1:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(img_path, 0)

        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32) / 255

        if ch == 1:
            x = torch.from_numpy(img[None])
        else:
            x = torch.from_numpy(img).permute(dims=(2, 0, 1))
        y = torch.tensor(label)
        return x, y

    def shuffle(self):
        random.shuffle(self.data_files)
