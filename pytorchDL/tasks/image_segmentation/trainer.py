import os
from multiprocessing import cpu_count

import torch
import numpy as np

from pytorchDL.trainer_base import TrainerBase
from pytorchDL.loggers import TensorboardLogger, ProgressLogger
from pytorchDL.dataset_iterator import DataIterator
from pytorchDL.networks.unet import UNet
from pytorchDL.tasks.image_segmentation.data import Dataset
from pytorchDL.utils.misc import generate_random_colors

from pytorchDL.metrics import MeanMetric


class Trainer(TrainerBase):

    def __init__(self, trainer_mode, out_dir, batch_size, max_epochs, train_steps_per_epoch, val_steps_per_epoch, log_interval,
                 init_lr, **cfg_args):

        super().__init__(trainer_mode, out_dir, batch_size, max_epochs, train_steps_per_epoch, val_steps_per_epoch, log_interval,
                         **cfg_args)

        class_weights = torch.Tensor(self.cfg['class_weights']).cuda()
        self.set_model(model=UNet(input_channels=self.cfg['input_size'][-1],
                                  output_channels=self.cfg['num_out_classes']),
                       device=torch.device('cuda'))

        self.set_optimizer(optimizer=torch.optim.Adam(params=self.model.parameters(), lr=init_lr))
        self.set_loss_fn(torch.nn.CrossEntropyLoss(weight=class_weights), device=torch.device('cuda'))

        self.setup()
        self._set_label_colors()
        self.save_config(os.path.join(self.out_dir, 'trainer_cfg.json'))

    def _set_label_colors(self):
        self.label_colors = generate_random_colors(self.cfg['num_out_classes'], random_seed=42).T

    def _proc_output_for_log(self, batch_data, y_pred):
        x, y = batch_data
        x = x.cpu().detach().numpy()[0]
        y = y.cpu().detach().numpy()[0]

        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        y_pred = y_pred.cpu().detach().numpy()[0]
        pred_labels = np.argmax(y_pred, axis=0)

        gt_label_img = self.label_colors[:, y]
        gt_label_img = gt_label_img[None]

        pred_label_img = self.label_colors[:, pred_labels]
        pred_label_img = pred_label_img[None]

        x = x[None]
        return x, gt_label_img, pred_label_img

    def train_on_batch(self, batch_data):
        self.optimizer.zero_grad()

        # forward pass
        x, y = batch_data
        y_pred = self.model(x.cuda())
        batch_loss = self.loss_fn(y_pred, y.cuda())

        # backward pass
        batch_loss.backward()
        self.optimizer.step()

        # logging
        batch_loss = batch_loss.item()
        self.ep_train_mean_loss(batch_loss)  # update mean epoch loss metric
        self.prog_logger.log(batch_loss=batch_loss, mean_loss=self.ep_train_mean_loss.result())

        if (self.state['train_step'] % self.log_interval) == 0:
            x, gt, pred = self._proc_output_for_log(batch_data, y_pred)
            log_data = [{'data': x, 'type': 'image', 'name': '0_input'},
                        {'data': gt, 'type': 'image', 'name': '1_gt_mask'},
                        {'data': pred, 'type': 'image', 'name': '2_pred_mask'},
                        {'data': batch_loss, 'type': 'scalar', 'name': 'batch_loss'}]

            self.tb_logger.log(log_data, stage='train', step=self.state['train_step'])

        self.state['train_step'] += 1  # update train step

    def eval_on_batch(self, batch_data):

        # forward pass
        x, y = batch_data
        y_pred = self.model(x.cuda())
        batch_loss = self.loss_fn(y_pred, y.cuda())

        # logging
        batch_loss = batch_loss.item()
        self.ep_val_mean_loss(batch_loss)  # update mean epoch loss metric
        self.prog_logger.log(batch_loss=batch_loss, mean_loss=self.ep_val_mean_loss.result())

        if (self.state['val_step'] % self.log_interval) == 0:
            x, gt, pred = self._proc_output_for_log(batch_data, y_pred)
            log_data = [
                {'data': x, 'type': 'image', 'name': '0_input'},
                {'data': gt, 'type': 'image', 'name': '1_gt_mask'},
                {'data': pred, 'type': 'image', 'name': '2_pred_mask'}]
            self.tb_logger.log(log_data, stage='val', step=self.state['val_step'])

        self.state['val_step'] += 1

    def _init_metrics(self):
        self.ep_train_mean_loss = MeanMetric()
        self.ep_val_mean_loss = MeanMetric()

    def _reset_metrics(self):
        self.ep_train_mean_loss.reset()
        self.ep_val_mean_loss.reset()

    def run(self):

        num_dataloader_workers = 0 if self.trainer_mode == 'debug' else cpu_count() - 2

        # load train and validation datasets iterators
        train_dataset = Dataset(data_dir=self.cfg['train_data_dir'], output_shape=self.cfg['input_size'])
        train_data_iterator = DataIterator(train_dataset, batch_size=self.batch_size,
                                           num_workers=num_dataloader_workers, shuffle=True)

        val_dataset = Dataset(data_dir=self.cfg['val_data_dir'], output_shape=self.cfg['input_size'])
        val_data_iterator = DataIterator(val_dataset, batch_size=self.batch_size,
                                         num_workers=num_dataloader_workers, shuffle=True)

        if self.train_steps_per_epoch <= 0:  # if train steps per epoch is <= 0, set it to cover the whole dataset
            self.train_steps_per_epoch = len(train_dataset) // self.batch_size

        if self.val_steps_per_epoch <= 0:
            self.val_steps_per_epoch = len(val_dataset) // self.batch_size

        self.tb_logger = TensorboardLogger(log_dir=os.path.join(self.log_dir, 'tensorboard'))
        self._init_metrics()

        for ep in range(self.state['epoch'], self.max_epochs):
            print('\nEPOCH: %d' % ep)
            self.state['epoch'] = ep

            self._reset_metrics()

            # TRAIN LOOP
            self.model.train()
            self.stage = 'train'
            self.prog_logger = ProgressLogger(total_steps=self.train_steps_per_epoch, description='Training')

            for i in range(self.train_steps_per_epoch):

                batch_data = next(train_data_iterator)
                self.train_on_batch(batch_data)

            self.tb_logger.log(log_data=[{'data': self.ep_train_mean_loss.result(), 'type': 'scalar',
                                          'name': '%s/ep_mean_loss' % self.stage, 'stage': self.stage}],
                               stage=self.stage, step=ep)

            self.prog_logger.close()
            self.save_checkpoint('checkpoint-step-%d' % self.state['train_step'])

            # VAL LOOP
            self.model.eval()
            self.stage = 'val'
            self.prog_logger = ProgressLogger(total_steps=self.val_steps_per_epoch, description='Validation')

            with torch.no_grad():
                for i in range(self.val_steps_per_epoch):

                    batch_data = next(val_data_iterator)
                    self.eval_on_batch(batch_data)

            self.tb_logger.log(log_data=[{'data': self.ep_val_mean_loss.result(), 'type': 'scalar',
                                          'name': '%s/ep_mean_loss' % self.stage, 'stage': self.stage}],
                               stage=self.stage,
                               step=ep)

            self.prog_logger.close()

            if self.ep_val_mean_loss.result() < self.state['best_val_loss']:
                print('\tMean validation loss decreased from %f to %f. Saving best model' % (self.state['best_val_loss'], self.ep_val_mean_loss.result()))
                self.state['best_val_loss'] = self.ep_val_mean_loss.result()
                self.save_best_checkpoint()

    def run_testing(self):
        raise NotImplementedError()
