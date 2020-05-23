import torch
import unittest

import numpy as np
from pytorchDL.networks.resnet import ResNet


class TestResnet(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.bs = 3  # batch size
        cls.input_batch_rgb = torch.rand(size=(cls.bs, 3, 64, 64))
        cls.input_batch_grayscale = torch.rand(size=(cls.bs, 1, 64, 64))

        cls.num_out_classes = np.random.randint(low=2, high=20)
        cls.y_target = torch.randint(0, cls.num_out_classes, size=[cls.bs])
        cls.ce_loss = torch.nn.CrossEntropyLoss()

    def test_grayscale_input(self):
        model = ResNet(input_size=(64, 64, 1),
                       num_out_classes=self.num_out_classes,
                       res_units_per_block=np.random.randint(1, 5))

        optim = torch.optim.Adam(params=model.parameters())

        output = model(self.input_batch_grayscale)
        self.assertTrue(output.size() == (self.bs, self.num_out_classes))
        loss = self.ce_loss(output, self.y_target)
        loss.backward()
        optim.step()

    def test_rgb_input(self):

        model = ResNet(input_size=(64, 64, 3),
                       num_out_classes=self.num_out_classes,
                       res_units_per_block=np.random.randint(1, 5))

        optim = torch.optim.Adam(params=model.parameters())

        output = model(self.input_batch_rgb)
        self.assertTrue(output.size() == (self.bs, self.num_out_classes))
        loss = self.ce_loss(output, self.y_target)
        loss.backward()
        optim.step()
