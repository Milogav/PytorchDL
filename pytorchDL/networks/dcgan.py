import torch
import os


def get_script_path():
    return os.path.realpath(__file__)


class Conv2dTransposeUnit(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, batch_norm=True, activation='relu', bias=False):
        super(Conv2dTransposeUnit, self).__init__()

        layers = [torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                           stride, padding=padding, output_padding=output_padding, bias=bias)]
        if batch_norm:
            layers.append(torch.nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=5e-05))

        if activation == 'relu':
            layers.append(torch.nn.ReLU(inplace=True))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class Conv2dUnit(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, batch_norm=True, activation='leaky_relu', bias=False):
        super(Conv2dUnit, self).__init__()

        layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=bias)]

        if batch_norm:
            layers.append(torch.nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=5e-05))

        if activation == 'leaky_relu':
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(torch.nn.ReLU(inplace=True))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class Generator(torch.nn.Module):

    def __init__(self, latent_vector_length, output_channels):
        super().__init__()
        self.dense = torch.nn.Linear(latent_vector_length, 8*8*1024)
        self.leaky_relu = torch.nn.LeakyReLU(0.2, inplace=True)

        self.ct1 = Conv2dTransposeUnit(1024, 512)
        self.ct2 = Conv2dTransposeUnit(512, 256)
        self.ct3 = Conv2dTransposeUnit(256, 128)
        self.ct4 = Conv2dTransposeUnit(128, 64)
        self.ct5 = Conv2dTransposeUnit(64, output_channels, stride=1, output_padding=0, batch_norm=False, activation=None)

    def forward(self, input_tensor):

        x = self.leaky_relu(self.dense(input_tensor))
        x = x.view(-1, 1024, 8, 8)
        x = self.ct1(x)
        x = self.ct2(x)
        x = self.ct3(x)
        x = self.ct4(x)
        x = self.ct5(x)
        x = torch.tanh(x)
        return x


class Discriminator(torch.nn.Module):

    def __init__(self, input_shape):
        super().__init__()

        self.cn1 = Conv2dUnit(input_shape[-1], 64, stride=2, activation='leaky_relu', batch_norm=False)
        self.cn2 = Conv2dUnit(64, 128, stride=2, activation='leaky_relu')
        self.cn3 = Conv2dUnit(128, 256, stride=2, activation='leaky_relu')
        self.cn4 = Conv2dUnit(256, 512, stride=2, activation='leaky_relu')
        self.cn5 = Conv2dUnit(512, 1024, stride=2, activation='leaky_relu')

        self.dense_size = int(1024 * input_shape[0] / 32 * input_shape[0] / 32)
        self.dense = torch.nn.Linear(self.dense_size, 1, bias=False)

    def forward(self, input_tensor):
        x = self.cn1(input_tensor)
        x = self.cn2(x)
        x = self.cn3(x)
        x = self.cn4(x)
        x = self.cn5(x)
        x = self.dense(x.view(-1, self.dense_size))
        return x
