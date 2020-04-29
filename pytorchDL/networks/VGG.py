import torch
import os


def get_script_path():
    return os.path.realpath(__file__)


class Conv2dUnit(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        super(Conv2dUnit, self).__init__()
        self._cn1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.relu = torch.nn.ReLU()

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


class VGG16(torch.nn.Module):

    def __init__(self, input_size, num_out_classes):
        super(VGG16, self).__init__()

        self.cn1_1 = Conv2dUnit(input_size[-1], 32)
        self.cn1_2 = Conv2dUnit(32, 32)
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn2_1 = Conv2dUnit(32, 64)
        self.cn2_1 = Conv2dUnit(64, 64)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn3_1 = Conv2dUnit(64, 128)
        self.cn3_2 = Conv2dUnit(128, 128)
        self.cn3_3 = Conv2dUnit(128, 128)
        self.mp3 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn4_1 = Conv2dUnit(128, 256)
        self.cn4_2 = Conv2dUnit(256, 256)
        self.cn4_3 = Conv2dUnit(256, 256)
        self.mp4 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn5_1 = Conv2dUnit(256, 256)
        self.cn5_2 = Conv2dUnit(256, 256)
        self.cn5_3 = Conv2dUnit(256, 256)
        self.mp5 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        r = 2 ** 5
        fc_input_size = 256 * input_size[1] // r * input_size[2] // r
        self.fc1 = FullyConnectedUnit(fc_input_size, 1024)
        self.drop1 = torch.nn.Dropout2d(p=0.5)

        self.fc2 = FullyConnectedUnit(1024, 512)
        self.drop2 = torch.nn.Dropout2d(p=0.5)

        self.fc3 = torch.nn.Linear(512, num_out_classes)

    def forward(self, input):
        x = self.cn1_1(input)
        x = self.cn1_2(x)
        x = self.mp1(x)

        x = self.cn2_1(x)
        x = self.cn2_1(x)
        x = self.mp2(x)

        x = self.cn3_1(x)
        x = self.cn3_2(x)
        x = self.cn3_3(x)
        x = self.mp3(x)

        x = self.cn4_1(x)
        x = self.cn4_2(x)
        x = self.cn4_3(x)
        x = self.mp4(x)

        x = self.cn5_1(x)
        x = self.cn5_2(x)
        x = self.cn5_3(x)
        x = self.mp5(x)

        x = self.fc1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)

        output = self.fc3(x)
        return output
