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


class ResidualUnit(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size=(3, 3), stride=(1, 1)):
        super(ResidualUnit, self).__init__()
        self._cn1 = torch.nn.Conv2d(in_features, out_features, kernel_size, stride, padding=1)
        self._bn1 = torch.nn.BatchNorm2d(out_features)

        self._cn2 = torch.nn.Conv2d(out_features, out_features, kernel_size, stride, padding=1)
        self._bn2 = torch.nn.BatchNorm2d(out_features)

        self.relu = torch.nn.ReLU()

    def forward(self, input):

        x = self._cn1(input)
        x = self._bn1(x)
        x = self.relu(x)

        x = self._cn2(x)
        x = self._bn2(x)

        out = torch.add(input, x)
        out = self.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, input_size, num_out_classes):
        super().__init__()

        self.cn0 = Conv2dUnit(input_size[-1], 32)

        self.cn1 = Conv2dUnit(32, 32, stride=2)
        self.res_block1 = torch.nn.Sequential(*[ResidualUnit(32, 32) for _ in range(3)])

        self.cn2 = Conv2dUnit(32, 64)
        self.res_block2 = torch.nn.Sequential(*[ResidualUnit(64, 64) for _ in range(3)])

        self.cn3 = Conv2dUnit(64, 128, stride=2)
        self.res_block3 = torch.nn.Sequential(*[ResidualUnit(128, 128) for _ in range(3)])

        self.cn4 = Conv2dUnit(128, 128)
        self.res_block4 = torch.nn.Sequential(*[ResidualUnit(128, 128) for _ in range(3)])

        self.cn5 = Conv2dUnit(128, 256, stride=2)
        self.res_block5 = torch.nn.Sequential(*[ResidualUnit(256, 256) for _ in range(3)])

        self.cn6 = Conv2dUnit(256, 512)

        self.fc1 = torch.nn.Linear(512, num_out_classes)

    def forward(self, input):
        x = self.cn0(input)

        x = self.cn1(x)
        x = self.res_block1(x)

        x = self.cn2(x)
        x = self.res_block2(x)

        x = self.cn3(x)
        x = self.res_block3(x)

        x = self.cn4(x)
        x = self.res_block4(x)

        x = self.cn5(x)
        x = self.res_block5(x)

        x = self.cn6(x)

        # global average pooling over channel dimension
        x = torch.mean(x, dim=[2, 3])

        out = self.fc1(x)
        return out
