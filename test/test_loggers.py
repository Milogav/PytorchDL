import os
from shutil import rmtree

import unittest
import torch

import numpy as np
from pytorchDL.loggers import TensorboardLogger


class TestTensorboardLogger(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cwd = os.path.dirname(os.path.realpath(__file__))
        cls.test_log_dir = os.path.join(cwd, 'test_logs')
        cls.tb_logger = TensorboardLogger(log_dir=cls.test_log_dir)

    def test_log_scalar(self):

        log_data = [{'type': 'scalar', 'data': 1.2342, 'name': 'set_0'},
                    {'type': 'scalar', 'data': np.random.uniform(), 'name': 'set_1'}]

        self.tb_logger.log(log_data, stage='train', step=0)
        self.tb_logger.log(log_data, stage='train', step=1)

        log_data = [{'type': 'scalar', 'data': torch.Tensor([0.1312]), 'name': 'set_1'}]
        self.tb_logger.log(log_data, stage='val', step=0)

    def test_log_np_image(self):
        img = np.random.uniform(0, 1, size=(1, 3, 32, 32))  # rgb image
        log_data = [{'type': 'image', 'data': img, 'name': 'imgs_0'}]
        self.tb_logger.log(log_data, stage='train', step=0)

        img = np.random.uniform(0, 1, size=(5, 1, 32, 32))  #grayscale image
        log_data = [{'type': 'image', 'data': img, 'name': 'imgs_0'}]
        self.tb_logger.log(log_data, stage='train', step=1)

    def test_log_tensor_image(self):

        img = torch.from_numpy(np.random.uniform(0, 1, size=(5, 3, 32, 32)))
        log_data = [{'type': 'image', 'data': img, 'name': 'imgs_0'},
                    {'type': 'image', 'data': img, 'name': 'imgs_1'}]

        self.tb_logger.log(log_data, stage='train', step=0)

    def test_log_np_pointcloud(self):
        pcd_xyz = np.random.uniform(-5, 5, size=(5, 3, 30))
        pcd_rgb = np.random.uniform(0, 1, size=(5, 3, 30))
        pcd = np.concatenate((pcd_xyz, pcd_rgb), axis=1)
        log_data = [{'type': 'pointcloud', 'data': pcd, 'name': 'pcd_0'}]

        self.tb_logger.log(log_data, stage='test', step=10)

    def test_log_tensor_pointcloud(self):
        pcd_xyz = torch.from_numpy(np.random.uniform(-5, 5, size=(5, 3, 30)))
        pcd_rgb = torch.from_numpy(np.random.uniform(0, 1, size=(5, 3, 30)))
        pcd = torch.cat((pcd_xyz, pcd_rgb), dim=1)
        log_data = [{'type': 'pointcloud', 'data': pcd, 'name': 'pcd_0'}]

        self.tb_logger.log(log_data, stage='test', step=10)

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(cls.test_log_dir)
