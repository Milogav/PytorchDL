import os
import json

import torch
from tqdm import tqdm

from pytorchDL.tasks.image_segmentation.predictor import Predictor
from pytorchDL.tasks.image_segmentation.data import Dataset
from pytorchDL.metrics import ConfusionMatrix


class Evaluator(Predictor):

    def __init__(self, test_data_dir, out_dir, ckpt_path, batch_size, device, num_proc, class_tags=None):
        super().__init__(ckpt_path, device, num_proc=num_proc)
        self.test_data_dir = test_data_dir
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.batch_size = batch_size
        self.class_tags = class_tags
        self.num_proc = num_proc

    def run_testing(self):
        test_dataset = Dataset(data_dir=self.test_data_dir, output_shape=self.cfg['input_size'])
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=self.batch_size,
                                                      num_workers=self.num_proc)

        cm = ConfusionMatrix(num_classes=self.cfg['num_out_classes'], tags=self.class_tags)
        test_steps = len(test_dataset) // self.batch_size
        test_results = {}
        with torch.no_grad():
            for batch_data in tqdm(test_dataloader, total=test_steps):

                x, y = batch_data
                x = x.to(self.device)

                pred_logits = self.model(x)
                pred_logits = torch.nn.functional.softmax(pred_logits, dim=1)
                _, pred_labels = pred_logits.max(dim=1)

                gt_labels = y.cpu().numpy().flatten()
                pred_labels = pred_labels.cpu().numpy().flatten()
                cm.update(gt_labels, pred_labels)

        out_file = os.path.join(self.out_dir, 'test_confusion_matrix.png')
        cm.plot(title='Conf. Matrix - Classification', normalized=True, to_file=out_file)

        test_results['norm_conf_mat'] = cm.get_normalized().tolist()
        test_results['class_tags'] = self.class_tags
        with open(os.path.join(self.out_dir, 'test_results.json'), 'w') as fp:
            json.dump(test_results, fp)


if __name__ == '__main__':

    test_dir = '/media/miguel/HDD/DeepLearning/Datasets/hand_landmark_detection/dataset_0/val'
    out_dir = '/home/miguel/prueba_hand_segmentation'
    ckpt_path = '/home/miguel/prueba_hand_segmentation/checkpoints/best_checkpoint.pth'

    class_tags = ['bckg', 'hand']
    evaluator = Evaluator(test_data_dir=test_dir,
                          out_dir=out_dir,
                          ckpt_path=ckpt_path,
                          batch_size=32,
                          device='gpu',
                          num_proc=0,
                          class_tags=class_tags)
    evaluator.run_testing()
