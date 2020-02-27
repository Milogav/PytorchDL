import torch
import os


def get_script_path():
    return os.path.realpath(__file__)


class Conv2dTransposeUnit(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1):
        super(Conv2dTransposeUnit, self).__init__()
        self.trcn1 = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                              stride, padding=padding, output_padding=output_padding)
        self.batch_norm = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.trcn1(input)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class Conv2dUnit(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        super(Conv2dUnit, self).__init__()
        self.cn1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.batch_norm = torch.nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = torch.nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input):
        x = self.cn1(input)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x


class Generator(torch.nn.Module):

    def __init__(self, latent_vector_length, output_channels):
        super().__init__()

        self.ct1 = Conv2dTransposeUnit(latent_vector_length, 512, kernel_size=4, stride=1, padding=0, output_padding=0)
        self.cn1 = Conv2dUnit(512, 512)

        self.ct2 = Conv2dTransposeUnit(512, 256, stride=2, padding=1, output_padding=1)
        self.cn2 = Conv2dUnit(256, 256)

        self.ct3 = Conv2dTransposeUnit(256, 128, stride=2, padding=1, output_padding=1)
        self.cn3 = Conv2dUnit(128, 128)

        self.ct4 = Conv2dTransposeUnit(128, 64, stride=2, padding=1, output_padding=1)
        self.cn4 = Conv2dUnit(64, 64)

        self.ct5 = Conv2dTransposeUnit(64, 32, stride=2, padding=1, output_padding=1)
        self.cn5 = Conv2dUnit(32, 32)

        self.ct6 = Conv2dTransposeUnit(32, 32, stride=2, padding=1, output_padding=1)
        self.cn6 = Conv2dUnit(32, 32)

        self.cn_last = torch.nn.Sequential(
            torch.nn.Conv2d(32, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.Tanh()
        )

    def forward(self, input_tensor):
        x = self.ct1(input_tensor)
        x = self.cn1(x)

        x = self.ct2(x)
        x = self.cn2(x)

        x = self.ct3(x)
        x = self.cn3(x)

        x = self.ct4(x)
        x = self.cn4(x)

        x = self.ct5(x)
        x = self.cn5(x)

        x = self.ct6(x)
        x = self.cn6(x)

        x = self.cn_last(x)
        return x


class Discriminator(torch.nn.Module):

    def __init__(self, input_shape):
        super().__init__()

        self.cn1 = Conv2dUnit(input_shape[-1], 32, stride=(2, 2))
        self.cn2 = Conv2dUnit(32, 64, stride=(2, 2))
        self.cn3 = Conv2dUnit(64, 128, stride=(2, 2))
        self.cn4 = Conv2dUnit(128, 128, stride=(2, 2))
        self.cn5 = Conv2dUnit(128, 256, stride=(2, 2))

        self.dense_size = int(256 * input_shape[0] / 32 * input_shape[0] / 32)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(self.dense_size, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, input_tensor):
        x = self.cn1(input_tensor)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.dense(x.view(-1, self.dense_size))
        return x
