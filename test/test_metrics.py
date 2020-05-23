import os
import unittest

import numpy as np
from pytorchDL.metrics import ConfusionMatrix, MeanMetric


class TestMeanMetric(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.mean_metric = MeanMetric()

    def test_result(self):
        self.mean_metric(0)
        self.mean_metric(1)
        self.assertTrue(self.mean_metric.result() == 0.5)

        self.mean_metric(0)
        self.mean_metric(1)
        self.assertTrue(self.mean_metric.result() == 0.5)


class TestConfusionMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.num_classes = np.random.randint(2, 10)
        cls.conf_mat = ConfusionMatrix(num_classes=cls.num_classes)

    def test_update(self):
        preds = np.array([0, 0, 1, 1, 0, 0, 1])
        truth = np.array([0, 0, 1, 1, 1, 1, 0])
        cm_test = np.zeros((self.num_classes, self.num_classes))
        cm_test[0, 0] = 2
        cm_test[1, 1] = 2
        cm_test[1, 0] = 2
        cm_test[0, 1] = 1

        self.conf_mat.update(truth, preds)
        self.assertTrue(np.all(self.conf_mat.cm == cm_test))

        self.conf_mat.update(truth, preds)
        self.assertTrue(np.all(self.conf_mat.cm == 2*cm_test))
        self.assertTrue(np.sum(self.conf_mat.cm) == len(preds)*2)
        self.conf_mat.reset()

    def test_perfect_predictions(self):

        preds = np.arange(0, self.num_classes)
        truth = preds

        self.conf_mat.update(truth, preds)
        self.conf_mat.update(truth, preds)

        norm_cm = self.conf_mat.get_normalized()

        self.assertTrue(np.all(norm_cm == np.eye(self.num_classes)))
        self.conf_mat.reset()

    def test_wrong_predictions(self):

        preds = np.random.randint(1, self.num_classes, 10)
        truth = preds - 1

        self.conf_mat.update(truth, preds)
        self.conf_mat.update(truth, preds)

        norm_cm = self.conf_mat.get_normalized()
        self.assertTrue(np.all(np.diag(norm_cm) == np.zeros(self.num_classes)))
        self.conf_mat.reset()

    def test_plot_to_file(self):

        preds = np.arange(0, self.num_classes)
        truth = preds

        self.conf_mat.update(truth, preds)
        self.conf_mat.update(truth, preds)

        self.conf_mat.plot(to_file='test_conf_mat.png')
        self.assertTrue(os.path.exists('test_conf_mat.png'))
        os.remove('test_conf_mat.png')
        self.conf_mat.reset()
