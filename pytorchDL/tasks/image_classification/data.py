import os
import glob

import cv2
import torch
import numpy as np

from pytorchDL.utils.imgproc import normalize


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, output_shape):
        """Data directory must contain one image directory per class, name after the class name.

        :param data_dir: directory containing the image and label data.
        :param output_shape: list or tuple defining the generator output shape
        """

        self.data_files = []
        self.class_tags = os.listdir(data_dir)

        for i, tag in enumerate(self.class_tags):
            class_dir = os.path.join(data_dir, tag)
            img_files = glob.glob(os.path.join(class_dir, '*'))
            self.data_files.append([path, i] for path in img_files)

        self.shuffle()
        self.output_shape = output_shape

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        img_path, label = self.data_files[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.output_shape[1], self.output_shape[0]))

        img = normalize(img, 0, 1)
        x = torch.tensor(img).permute(dims=(2, 0, 1)).type(torch.FloatTensor)
        y = torch.tensor([label])
        return x, y

    def shuffle(self):
        self.data_files = np.random.permutation(self.data_files)
