import os
import json

import torch
from tqdm import tqdm

from pytorchDL.tasks.image_classification.predictor import Predictor
from pytorchDL.tasks.image_classification.data import Dataset
from pytorchDL.metrics import ConfusionMatrix


class Evaluator(Predictor):

    def __init__(self, test_data_dir, out_dir, ckpt_path, batch_size, device, num_proc):
        super().__init__(ckpt_path, device, num_proc=num_proc)
        self.test_data_dir = test_data_dir
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.batch_size = batch_size

    def run_testing(self):
        test_dataset = Dataset(data_dir=self.test_data_dir, output_shape=self.cfg['input_size'])
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=self.batch_size,
                                                      num_workers=self.num_proc)

        cm = ConfusionMatrix(num_classes=self.cfg['num_out_classes'], tags=test_dataset.class_tags)
        test_steps = len(test_dataset) // self.batch_size
        test_results = {}
        with torch.no_grad():
            for batch_data in tqdm(test_dataloader, total=test_steps):

                x, y = batch_data
                x = x.to(self.device)

                pred_logits = self.model(x)
                pred_logits = torch.nn.functional.softmax(pred_logits, dim=1)
                _, pred_labels = pred_logits.max(dim=1)

                gt_labels = y.cpu().numpy()
                pred_labels = pred_labels.cpu().numpy()
                cm.update(gt_labels, pred_labels)

        out_file = os.path.join(self.out_dir, 'test_confusion_matrix.png')
        cm.plot(title='Conf. Matrix - Classification', normalized=True, to_file=out_file)

        test_results['norm_conf_mat'] = cm.get_normalized().tolist()
        test_results['class_tags'] = test_dataset.class_tags
        with open(os.path.join(self.out_dir, 'test_results.json'), 'w') as fp:
            json.dump(test_results, fp)


if __name__ == '__main__':

    test_dir = '/media/miguel/HDD/DeepLearning/Datasets/kaggle_mnist/val'
    out_dir = '/home/miguel/prueba_clasificacion_mnist/exp_0'
    ckpt_path = '/home/miguel/prueba_clasificacion_mnist/exp_0/checkpoints/best_checkpoint.pth'

    evaluator = Evaluator(test_data_dir=test_dir,
                          out_dir=out_dir,
                          ckpt_path=ckpt_path,
                          batch_size=32,
                          device='gpu',
                          num_proc=4)
    evaluator.run_testing()
