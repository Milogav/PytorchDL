import torch
import os
from pytorchDL.layers import Conv2dLayer, FullyConnectedLayer


def get_script_path():
    return os.path.realpath(__file__)


class VGG11(torch.nn.Module):

    def __init__(self, input_size, num_out_classes, **kwargs):
        super(VGG11, self).__init__()

        self.cn1 = Conv2dLayer(input_size[-1], 32, activation='relu')
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn2 = Conv2dLayer(self.cn1.out_features, 64, activation='relu')
        self.mp2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn3_1 = Conv2dLayer(self.cn2.out_features, 128, activation='relu')
        self.cn3_2 = Conv2dLayer(self.cn3_1.out_features, 128, activation='relu')
        self.mp3 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn4_1 = Conv2dLayer(self.cn3_2.out_features, 128, activation='relu')
        self.cn4_2 = Conv2dLayer(self.cn4_1.out_features, 128, activation='relu')

        self.cn5_1 = Conv2dLayer(self.cn4_2.out_features, 256, activation='relu')
        self.cn5_2 = Conv2dLayer(self.cn5_1.out_features, 256, activation='relu')

        self.drop1 = torch.nn.Dropout(p=0.5)

        r = 2 ** 3
        fc1_input_size = self.cn5_2.out_features * input_size[0] // r * input_size[1] // r
        self.fc1 = FullyConnectedLayer(fc1_input_size, 256, activation='relu')

        self.drop2 = torch.nn.Dropout(p=0.5)
        self.fc2 = FullyConnectedLayer(self.fc1.out_features, 128, activation='relu')

        self.fc3 = FullyConnectedLayer(self.fc2.out_features, num_out_classes, activation=None)

    def forward(self, input_tensor):
        x = self.cn1(input_tensor)
        x = self.mp1(x)

        x = self.cn2(x)
        x = self.mp2(x)

        x = self.cn3_1(x)
        x = self.cn3_2(x)
        x = self.mp3(x)

        x = self.cn4_1(x)
        x = self.cn4_2(x)

        x = self.cn5_1(x)
        x = self.cn5_2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)

        output = self.fc3(x)
        return output
