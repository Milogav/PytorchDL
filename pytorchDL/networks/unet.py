import os
import torch
from pytorchDL.layers import Conv2dLayer, Conv2dTransposeLayer


def get_script_path():
    return os.path.realpath(__file__)


class Conv2dBlock(torch.nn.Module):

    def __init__(self, in_features, out_features, kernel_size=(3, 3), stride=(1, 1)):
        super(Conv2dBlock, self).__init__()

        self.out_features = out_features
        self._cn1 = Conv2dLayer(in_features, out_features, activation='relu',
                                kernel_size=kernel_size, stride=stride, padding=1)

        self._cn2 = Conv2dLayer(out_features, out_features, activation='relu',
                                kernel_size=kernel_size, stride=stride, padding=1)

    def forward(self, input_tensor):
        x = self._cn1(input_tensor)
        x = self._cn2(x)
        return x


class UNet(torch.nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super(UNet, self).__init__()

        self.cn1 = Conv2dBlock(input_channels, 32)
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn2 = Conv2dBlock(32, 64)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn3 = Conv2dBlock(64, 128)
        self.mp3 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn4 = Conv2dBlock(128, 256)
        self.mp4 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.drop1 = torch.nn.Dropout2d(p=0.3)
        self.cn5 = Conv2dBlock(256, 512)

        self.up_cn1 = Conv2dTransposeLayer(512, 256, activation='relu')
        self.cn6 = Conv2dLayer(512, 256, activation='relu')

        self.up_cn2 = Conv2dTransposeLayer(256, 128, activation='relu')
        self.cn7 = Conv2dLayer(256, 128, activation='relu')

        self.up_cn3 = Conv2dTransposeLayer(128, 64, activation='relu')
        self.cn8 = Conv2dLayer(128, 64, activation='relu')

        self.up_cn4 = Conv2dTransposeLayer(64, 32, activation='relu')
        self.cn9 = Conv2dLayer(64, 32, activation='relu')

        self.cn10 = Conv2dLayer(32, output_channels, activation=None, kernel_size=(3, 3), padding=1)

    def forward(self, input_tensor):

        x1 = self.cn1(input_tensor)
        dw_x1 = self.mp1(x1)

        x2 = self.cn2(dw_x1)
        dw_x2 = self.mp2(x2)

        x3 = self.cn3(dw_x2)
        dw_x3 = self.mp3(x3)

        x4 = self.cn4(dw_x3)
        dw_x4 = self.mp4(x4)

        dw_x4 = self.drop1(dw_x4)
        x5 = self.cn5(dw_x4)

        up_x5 = self.up_cn1(x5)
        x6 = torch.cat((x4, up_x5), dim=1)
        x6 = self.cn6(x6)

        up_x6 = self.up_cn2(x6)
        x7 = torch.cat((x3, up_x6), dim=1)
        x7 = self.cn7(x7)

        up_x7 = self.up_cn3(x7)
        x8 = torch.cat((x2, up_x7), dim=1)
        x8 = self.cn8(x8)

        up_x8 = self.up_cn4(x8)
        x9 = torch.cat((x1, up_x8), dim=1)
        x9 = self.cn9(x9)

        return self.cn10(x9)
