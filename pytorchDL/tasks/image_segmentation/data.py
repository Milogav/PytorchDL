import os
import glob

import cv2
import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, output_shape):
        """Data directory must an "images" folder with the images and a "masks" folder with the corresponding label
         masks. Corresponding image and mask files must share the same filename and extension.

        :param data_dirs: directory containing the image and label data.
        :param output_shape: list or tuple defining the generator output shape
        """

        self.data_files = []
        img_dir = os.path.join(data_dir, 'images')
        mask_dir = os.path.join(data_dir, 'masks')
        img_files = glob.glob(os.path.join(img_dir, '*'))
        for img_path in img_files:
            mask_path = img_path.replace(img_dir, mask_dir)
            if os.path.exists(mask_path):
                self.data_files.append([img_path, mask_path])

        self.shuffle()
        self.output_shape = output_shape

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):

        h, w, ch = self.output_shape
        if ch > 1:
            img = cv2.imread(self.data_files[index][0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(self.data_files[index][0], 0)
            img = np.expand_dims(img, axis=2)

        labels = cv2.imread(self.data_files[index][1], 0)

        img = cv2.resize(img, (w, h))
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255

        if ch == 1:
            x = torch.from_numpy(img[None])
        else:
            x = torch.from_numpy(img).permute(dims=(2, 0, 1))

        y = torch.tensor(labels).type(torch.long)
        return x, y

    def shuffle(self):
        self.data_files = np.random.permutation(self.data_files)
