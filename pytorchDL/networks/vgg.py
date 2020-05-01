import torch
import os


def get_script_path():
    return os.path.realpath(__file__)


class Conv2dUnit(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        super(Conv2dUnit, self).__init__()
        self._cn1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.relu = torch.nn.ReLU()
        self.out_channels = out_channels

    def forward(self, input):
        x = self._cn1(input)
        x = self.relu(x)
        return x


class FullyConnectedUnit(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(FullyConnectedUnit, self).__init__()
        self._linear = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        x = self._linear(input)
        x = self.relu(x)
        return x


class VGG11(torch.nn.Module):

    def __init__(self, input_size, num_out_classes):
        super(VGG11, self).__init__()

        self.cn1= Conv2dUnit(input_size[-1], 32)
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn2 = Conv2dUnit(32, 64)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn3_1 = Conv2dUnit(64, 128)
        self.cn3_2 = Conv2dUnit(128, 128)
        self.mp3 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn4_1 = Conv2dUnit(128, 128)
        self.cn4_2 = Conv2dUnit(128, 128)

        self.cn5_1 = Conv2dUnit(128, 256)
        self.cn5_2 = Conv2dUnit(256, 256)

        self.drop1 = torch.nn.Dropout(p=0.5)

        r = 2 ** 3
        fc1_input_size = self.cn5_2.out_channels * input_size[0] // r * input_size[1] // r
        self.fc1 = FullyConnectedUnit(fc1_input_size, 256)

        self.drop2 = torch.nn.Dropout(p=0.5)
        self.fc2 = FullyConnectedUnit(256, 128)

        self.fc3 = torch.nn.Linear(128, num_out_classes)

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
