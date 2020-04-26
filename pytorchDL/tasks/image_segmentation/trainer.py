import os
from shutil import copy
from multiprocessing import cpu_count

import torch
import numpy as np

from pytorchDL.trainer_base import TrainerBase
from pytorchDL.loggers import TensorboardLogger, ProgressLogger
from pytorchDL.dataset_iterator import DataIterator
from pytorchDL.networks.unet import UNet
from pytorchDL.tasks.image_segmentation.data import Dataset

from pytorchDL.utils.metrics import MeanMetric


class Trainer(TrainerBase):

    def do_backup(self):

        copy(os.path.realpath(__file__), self.cfg['out_dir'])

    def setup(self, mode, train_data_dir, val_data_dir, input_shape=None, num_classes=None, init_lr=0.001, class_weights=None):

        self.do_backup()
        self.mode = mode
        self.cfg['train_data_dir'] = train_data_dir
        self.cfg['val_data_dir'] = val_data_dir

        self.cfg['input_shape'] = input_shape
        self.cfg['num_classes'] = num_classes

        if class_weights is None:
            class_weights = np.ones(num_classes)

        self.cfg['class_weights'] = class_weights

        # initialize model, optimizer and loss function
        self.model = UNet(input_channels=self.cfg['input_shape'][-1],
                          output_channels=self.cfg['num_classes'])
        self.model.cuda()

        self._set_label_colors()

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=init_lr)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).cuda())

        if mode == 'start':
            if len(os.listdir(self.cfg['checkpoint_dir'])) > 0:
                raise Exception('Error! Output checkpoint dir (%s) not empty, which is incompatible with "%s" trainer mode'
                                % (self.cfg['checkpoint_dir'], mode))
        elif mode == 'resume':
            if not len(os.listdir(self.cfg['checkpoint_dir'])):
                raise Exception('Error! Cannot resume from an empty checkpoint dir. Use "start" trainer mode instead')
            self.load_last_checkpoint(self.cfg['checkpoint_dir'])
        elif mode == 'debug':
            pass
        else:
            raise Exception('Error! Input trainer mode (%s) not available' % mode)

    def _set_label_colors(self):
        np.random.seed(42)
        self.label_colors = np.random.uniform(0, 1, size=(3, self.cfg['num_classes']))
        np.random.seed()

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

        if (self.state['train_step'] % self.cfg['log_interval']) == 0:
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

        if (self.state['val_step'] % self.cfg['log_interval']) == 0:
            x, gt, pred = self._proc_output_for_log(batch_data, y_pred)
            log_data = [
                {'data': x, 'type': 'image', 'name': '0_input'},
                {'data': gt, 'type': 'image', 'name': '1_gt_mask'},
                {'data': pred, 'type': 'image', 'name': '2_pred_mask'}]
            self.tb_logger.log(log_data, stage='val', step=self.state['val_step'])

        self.state['val_step'] += 1

    def run(self):

        if self.mode == 'debug':
            num_dataloader_workers = 0
        else:
            num_dataloader_workers = cpu_count() - 2

        # load train and validation datasets iterators
        train_dataset = Dataset(data_dir=self.cfg['train_data_dir'], output_shape=self.cfg['input_shape'])
        train_data_iterator = DataIterator(train_dataset, batch_size=self.cfg['batch_size'],
                                           num_workers=num_dataloader_workers, shuffle=True)

        val_dataset = Dataset(data_dir=self.cfg['val_data_dir'], output_shape=self.cfg['input_shape'])
        val_data_iterator = DataIterator(val_dataset, batch_size=self.cfg['batch_size'],
                                         num_workers=num_dataloader_workers, shuffle=True)

        if self.cfg['train_steps_per_epoch'] <= 0:  # if train steps per epoch is <= 0, set it to cover the whole dataset
            self.cfg['train_steps_per_epoch'] = len(train_dataset) // self.cfg['batch_size']

        if self.cfg['val_steps_per_epoch'] <= 0:
            self.cfg['val_steps_per_epoch'] = len(val_dataset) // self.cfg['batch_size']

        self.tb_logger = TensorboardLogger(log_dir=os.path.join(self.cfg['log_dir'], 'tensorboard'))
        self.ep_train_mean_loss = MeanMetric()
        self.ep_val_mean_loss = MeanMetric()

        for ep in range(self.state['epoch'], self.cfg['max_epochs']):
            print('\nEPOCH: %d' % ep)
            self.state['epoch'] = ep

            self.ep_train_mean_loss.reset()
            self.ep_val_mean_loss.reset()

            # TRAIN LOOP
            self.model.train()
            self.stage = 'train'
            self.prog_logger = ProgressLogger(total_steps=self.cfg['train_steps_per_epoch'], description='Training')

            for i in range(self.cfg['train_steps_per_epoch']):

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
            self.prog_logger = ProgressLogger(total_steps=self.cfg['val_steps_per_epoch'], description='Validation')

            with torch.no_grad():
                for i in range(self.cfg['val_steps_per_epoch']):

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
                self.save_checkpoint('best_checkpoint')
