import torch
import os


def get_script_path():
    return os.path.realpath(__file__)


class Conv2dUnit(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        super(Conv2dUnit, self).__init__()
        self._cn1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.relu = torch.nn.ReLU()
        self._cn2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=1)

    def forward(self, input):
        x = self._cn1(input)
        x = self.relu(x)
        x = self._cn2(x)
        x = self.relu(x)
        return x


class Conv2dTransposeUnit(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2)):
        super(Conv2dTransposeUnit, self).__init__()
        self._trcn1 = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        x = self._trcn1(input)
        x = self.relu(x)
        return x


class UNet(torch.nn.Module):

    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()

        self.cn1 = Conv2dUnit(input_channels, 32)
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn2 = Conv2dUnit(32, 64)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn3 = Conv2dUnit(64, 128)
        self.mp3 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.cn4 = Conv2dUnit(128, 256)
        self.mp4 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.drop1 = torch.nn.Dropout2d(p=0.3)
        self.cn5 = Conv2dUnit(256, 512)

        self.upcn1 = Conv2dTransposeUnit(512, 256)
        self.cn6 = Conv2dUnit(512, 256)

        self.upcn2 = Conv2dTransposeUnit(256, 128)
        self.cn7 = Conv2dUnit(256, 128)

        self.upcn3 = Conv2dTransposeUnit(128, 64)
        self.cn8 = Conv2dUnit(128, 64)

        self.upcn4 = Conv2dTransposeUnit(64, 32)
        self.cn9 = Conv2dUnit(64, 32)

        self.cn10 = torch.nn.Conv2d(32, output_channels, kernel_size=(3, 3), padding=1)

    def forward(self, input):

        x1 = self.cn1(input)
        dw_x1 = self.mp1(x1)

        x2 = self.cn2(dw_x1)
        dw_x2 = self.mp2(x2)

        x3 = self.cn3(dw_x2)
        dw_x3 = self.mp3(x3)

        x4 = self.cn4(dw_x3)
        dw_x4 = self.mp4(x4)

        dw_x4 = self.drop1(dw_x4)
        x5 = self.cn5(dw_x4)

        up_x5 = self.upcn1(x5)
        x6 = torch.cat((x4, up_x5), dim=1)
        x6 = self.cn6(x6)

        up_x6 = self.upcn2(x6)
        x7 = torch.cat((x3, up_x6), dim=1)
        x7 = self.cn7(x7)

        up_x7 = self.upcn3(x7)
        x8 = torch.cat((x2, up_x7), dim=1)
        x8 = self.cn8(x8)

        up_x8 = self.upcn4(x8)
        x9 = torch.cat((x1, up_x8), dim=1)
        x9 = self.cn9(x9)

        return self.cn10(x9)
