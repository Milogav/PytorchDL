import torch


class ActivationLayer(torch.nn.Module):

    def __init__(self, activation, **kwargs):
        super(ActivationLayer, self).__init__()

        activation = activation.lower()
        if activation == 'relu':
            self._act = torch.nn.ReLU(**kwargs)
        elif activation == 'leaky_relu':
            self._act = torch.nn.LeakyReLU(**kwargs)
        else:
            raise NotImplementedError('Error: %s activation layer not implemented!' % activation.upper())

    def forward(self, input_tensor):
        return self._act(input_tensor)


class Conv2dLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, activation, bn=False, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True):
        super(Conv2dLayer, self).__init__()
        
        self.out_features = out_features
        layers = list()
        layers.append(torch.nn.Conv2d(in_features, out_features, kernel_size, stride, padding=padding, bias=bias))
        if bn:
            layers.append(torch.nn.BatchNorm2d(num_features=out_features))
        if activation is not None:
            layers.append(ActivationLayer(activation))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input_tensor):
        return self.layers(input_tensor)


class Conv2dTransposeLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, activation, bn=False, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1, bias=True):
        super(Conv2dTransposeLayer, self).__init__()

        self.out_features = out_features
        layers = list()
        layers.append(torch.nn.ConvTranspose2d(in_features, out_features, kernel_size, stride, padding=padding, output_padding=output_padding, bias=bias))
        if bn:
            layers.append(torch.nn.BatchNorm2d(num_features=out_features))
        if activation is not None:
            layers.append(ActivationLayer(activation))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input_tensor):
        return self.layers(input_tensor)


class FullyConnectedLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, activation, bn=False, bias=True):
        super(FullyConnectedLayer, self).__init__()

        self.out_features = out_features
        layers = list()
        layers.append(torch.nn.Linear(in_features, out_features, bias=bias))
        if bn:
            layers.append(torch.nn.BatchNorm1d(num_features=out_features))
        if activation is not None:
            layers.append(ActivationLayer(activation))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input_tensor):
        return self.layers(input_tensor)


class ResidualConv2dLayer(torch.nn.Module):
    def __init__(self, in_features, activation, bn=False, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True):

        super(ResidualConv2dLayer, self).__init__()

        self.out_features = in_features
        self._cn0 = Conv2dLayer(in_features, in_features,
                                activation=activation, bn=bn, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)

        self._cn1 = Conv2dLayer(in_features, in_features,
                                activation=None, bn=bn, kernel_size=kernel_size,
                                stride=1, padding=padding, bias=bias)

        self.act = ActivationLayer(activation) if activation is not None else None

    def forward(self, input_tensor):

        x = self._cn0(input_tensor)
        x = self._cn1(x)

        out = torch.add(input_tensor, x)
        if self.act is not None:
            out = self.act(out)
        return out
