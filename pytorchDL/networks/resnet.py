import os
import torch
from pytorchDL.layers import ResidualConv2dLayer, FullyConnectedLayer, Conv2dLayer


def get_script_path():
    return os.path.realpath(__file__)


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features, n_res_units, bn=True):
        super(ResidualBlock, self).__init__()

        self.out_features = in_features

        res_units = list()
        res_units.append(ResidualConv2dLayer(in_features, activation='relu', bn=bn))
        for _ in range(1, n_res_units):
            res_units.append(ResidualConv2dLayer(in_features, activation='relu', bn=bn))

        self._res_block = torch.nn.Sequential(*res_units)

    def forward(self, input_tensor):
        return self._res_block(input_tensor)


class ResNet(torch.nn.Module):
    def __init__(self, input_size, num_out_classes, res_units_per_block=2, **kwargs):
        super().__init__()

        self.cn0 = Conv2dLayer(input_size[-1], 32, activation='relu')

        self.cn1 = Conv2dLayer(32, 32, activation='relu', stride=2)
        self.res_block1 = ResidualBlock(32, n_res_units=res_units_per_block)

        self.cn2 = Conv2dLayer(32, 64, activation='relu')
        self.res_block2 = ResidualBlock(64, n_res_units=res_units_per_block)

        self.cn3 = Conv2dLayer(64, 128, activation='relu', stride=2)
        self.res_block3 = ResidualBlock(128, n_res_units=res_units_per_block)

        self.cn4 = Conv2dLayer(128, 128, activation='relu')
        self.res_block4 = ResidualBlock(128, n_res_units=res_units_per_block)

        self.cn5 = Conv2dLayer(128, 256, activation='relu', stride=2)
        self.res_block5 = ResidualBlock(256, n_res_units=res_units_per_block)

        self.cn6 = Conv2dLayer(256, 512, activation='relu')

        self.fc1 = FullyConnectedLayer(512, num_out_classes, activation=None)

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
