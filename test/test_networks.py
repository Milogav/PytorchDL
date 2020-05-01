import torch
import unittest

import numpy as np
from pytorchDL.networks.resnet import ResNet
from pytorchDL.networks.unet import UNet
from pytorchDL.networks.vgg import VGG11


class TestNetworks(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.bs = 3  # batch size
        cls.input_batch_rgb = torch.rand(size=(cls.bs, 3, 64, 64))
        cls.input_batch_grayscale = torch.rand(size=(cls.bs, 1, 64, 64))
        cls.ce_loss = torch.nn.CrossEntropyLoss()

    def test_resnet(self):

        num_out_classes = np.random.randint(low=2, high=20)
        y_target = torch.randint(0, num_out_classes, size=[self.bs])

        # test forward and backward pass on RGB input
        model = ResNet(input_size=(64, 64, 3), num_out_classes=num_out_classes)
        optim = torch.optim.Adam(params=model.parameters())

        output = model(self.input_batch_rgb)
        self.assertTrue(output.size() == (self.bs, num_out_classes))
        loss = self.ce_loss(output, y_target)
        loss.backward()
        optim.step()

        # test forward and backward pass on grayscale input
        model = ResNet(input_size=(64, 64, 1), num_out_classes=num_out_classes)
        optim = torch.optim.Adam(params=model.parameters())

        output = model(self.input_batch_grayscale)
        self.assertTrue(output.size() == (self.bs, num_out_classes))
        loss = self.ce_loss(output, y_target)
        loss.backward()
        optim.step()

    def test_vgg11(self):
        num_out_classes = np.random.randint(low=2, high=20)
        y_target = torch.randint(0, num_out_classes, size=[self.bs])

        # test forward and backward pass on RGB input
        model = VGG11(input_size=(64, 64, 3), num_out_classes=num_out_classes)
        optim = torch.optim.Adam(params=model.parameters())

        output = model(self.input_batch_rgb)
        self.assertTrue(output.size() == (self.bs, num_out_classes))
        loss = self.ce_loss(output, y_target)
        loss.backward()
        optim.step()

        # test forward and backward pass on grayscale input
        model = VGG11(input_size=(64, 64, 1), num_out_classes=num_out_classes)
        optim = torch.optim.Adam(params=model.parameters())

        output = model(self.input_batch_grayscale)
        self.assertTrue(output.size() == (self.bs, num_out_classes))
        loss = self.ce_loss(output, y_target)
        loss.backward()
        optim.step()

    def test_unet(self):

        num_out_labels = np.random.randint(low=1, high=10)
        y_target = torch.randint(0, num_out_labels, size=[self.bs, 64, 64])

        # test forward and backward pass on RGB input
        model = UNet(input_channels=3, output_channels=num_out_labels)
        optim = torch.optim.Adam(params=model.parameters())

        output = model(self.input_batch_rgb)
        self.assertTrue(output.size() == (self.bs, num_out_labels, 64, 64))
        loss = self.ce_loss(output, y_target)
        loss.backward()
        optim.step()

        # test forward and backward pass on grayscale input
        model = UNet(input_channels=1, output_channels=num_out_labels)
        optim = torch.optim.Adam(params=model.parameters())

        output = model(self.input_batch_grayscale)
        self.assertTrue(output.size() == (self.bs, num_out_labels, 64, 64))
        loss = self.ce_loss(output, y_target)
        loss.backward()
        optim.step()
