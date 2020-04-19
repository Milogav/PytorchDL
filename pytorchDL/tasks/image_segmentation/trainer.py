import os
from shutil import copy
from multiprocessing import cpu_count

import torch
import numpy as np

from pytorchDL.trainer_base import TrainerBase
from pytorchDL.networks.unet import UNet
from pytorchDL.tasks.image_segmentation.data import Dataset
from pytorchDL.utils.misc import Timer, print_in_place
from pytorchDL.utils.metrics import MeanMetric


class Trainer(TrainerBase):

    def do_backup(self):

        copy(os.path.realpath(__file__), self.cfg['out_dir'])

    def setup(self, train_mode, train_data_dir, val_data_dir, input_shape=None, num_classes=None, init_lr=0.001, class_weights=None):

        self.do_backup()
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

        if train_mode == 'start':
            if len(os.listdir(self.cfg['checkpoint_dir'])) > 0:
                raise Exception('Error! Output checkpoint dir (%s) not empty, which is incompatible with "%s" training mode'
                                % (self.cfg['checkpoint_dir'], train_mode))
        elif train_mode == 'resume':
            if not len(os.listdir(self.cfg['checkpoint_dir'])):
                raise Exception('Error! Cannot resume from an empty checkpoint dir. Use "start" train mode instead')
            self.load_last_checkpoint(self.cfg['checkpoint_dir'])
        elif train_mode == 'debug':
            pass
        else:
            raise Exception('Error! Input train mode (%s) not available' % train_mode)

    def _load_train_dataset(self, num_workers):
        train_dataset = Dataset(data_dir=self.cfg['train_data_dir'],
                                output_shape=self.cfg['input_shape'])

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=self.cfg['batch_size'],
                                                       shuffle=False,
                                                       num_workers=num_workers)
        self.num_train_examples = len(train_dataset)
        self.train_dataset_iterator = iter(train_dataloader)

    def _load_val_dataset(self, num_workers):
        val_dataset = Dataset(data_dir=self.cfg['val_data_dir'],
                              output_shape=self.cfg['input_shape'])

        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=self.cfg['batch_size'],
                                                     shuffle=True,
                                                     num_workers=num_workers)

        self.num_val_examples = len(val_dataset)
        self.val_dataset_iterator = iter(val_dataloader)

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

        # define forward pass from batch_data
        x, y = batch_data
        y_pred = self.model(x.cuda())
        batch_loss = self.loss_fn(y_pred, y.cuda())

        batch_loss.backward()
        self.optimizer.step()
        self.state['train_step'] += 1

        batch_output = {'batch_loss': batch_loss.item(),
                        'predictions': y_pred}

        return batch_output

    def eval_on_batch(self, batch_data):

        x, y = batch_data
        y_pred = self.model(x.cuda())
        batch_loss = self.loss_fn(y_pred, y.cuda())
        self.state['val_step'] += 1

        batch_output = {'batch_loss': batch_loss.item(),
                        'predictions': y_pred}

        return batch_output

    def run(self):

        num_dataloader_workers = cpu_count() - 2

        # load train and validation datasets iterators
        self._load_train_dataset(num_workers=num_dataloader_workers)
        self._load_val_dataset(num_workers=num_dataloader_workers)

        if self.cfg['train_steps_per_epoch'] <= 0:  # if train steps per epoch is <= 0, set it to cover the whole dataset
            self.cfg['train_steps_per_epoch'] = self.num_train_examples // self.cfg['batch_size']

        if self.cfg['val_steps_per_epoch'] <= 0:
            self.cfg['val_steps_per_epoch'] = self.num_val_examples // self.cfg['batch_size']

        self.create_tensorboard_summary(launch_tensorboard=True)
        for ep in range(self.state['epoch'], self.cfg['max_epochs']):
            print('\n\nTRAINING EPOCH: %d' % ep)
            self.state['epoch'] = ep

            # TRAIN LOOP
            self.model.train()
            self.stage = 'train'
            ep_train_mean_loss = MeanMetric()
            timer = Timer(total_steps=self.cfg['train_steps_per_epoch'])
            for i in range(self.cfg['train_steps_per_epoch']):
                try:
                    batch_data = next(self.train_dataset_iterator)
                except StopIteration:  # when dataset is exhausted, reload it from scratch
                    print('\nTrain dataset exhausted. Reloading dataset...')
                    self._load_train_dataset(num_workers=num_dataloader_workers)
                    batch_data = next(self.train_dataset_iterator)

                batch_output = self.train_on_batch(batch_data)
                ep_train_mean_loss(batch_output['batch_loss'])

                if (i % self.cfg['log_interval']) == 0:
                    x, gt, pred = self._proc_output_for_log(batch_data, batch_output['predictions'])
                    log_data = [{'data': x, 'type': 'image', 'name': '%s/0_input' % self.stage, 'stage': self.stage},
                                {'data': gt, 'type': 'image', 'name': '%s/1_gt_mask' % self.stage, 'stage': self.stage},
                                {'data': pred, 'type': 'image', 'name': '%s/2_pred_mask' % self.stage, 'stage': self.stage},
                                {'data': batch_output['batch_loss'], 'type': 'scalar', 'name': '%s/batch_loss' % self.stage, 'stage': self.stage}]
                    self.log_to_tensorboard(log_data)

                time_left, it_time = timer()
                print_in_place('Epoch: %d (training) -- Train step: %d -- Batch: %d/%d -- ETA: %s -- It. time: %.4f s -- Mean loss: %f'
                               % (ep, self.state['train_step'], i, self.cfg['train_steps_per_epoch'], time_left, it_time, ep_train_mean_loss.result()))

            self.log_to_tensorboard([{'data': ep_train_mean_loss.result(), 'type': 'scalar',
                                      'name': '%s/ep_mean_loss' % self.stage, 'stage': self.stage}],
                                    step=ep)

            print('')
            self.save_checkpoint('checkpoint-step-%d' % self.state['train_step'])

            # VAL LOOP
            self.model.eval()
            self.stage = 'val'
            ep_val_mean_loss = MeanMetric()
            timer = Timer(total_steps=self.cfg['val_steps_per_epoch'])
            with torch.no_grad():
                for i in range(self.cfg['val_steps_per_epoch']):
                    try:
                        batch_data = next(self.val_dataset_iterator)
                    except StopIteration:
                        print('\nVal dataset exhausted. Reloading dataset...')
                        self._load_val_dataset(num_workers=num_dataloader_workers)
                        batch_data = next(self.val_dataset_iterator)

                    batch_output = self.eval_on_batch(batch_data)

                    ep_val_mean_loss(batch_output['batch_loss'])

                    if (i % self.cfg['log_interval']) == 0:
                        x, gt, pred = self._proc_output_for_log(batch_data, batch_output['predictions'])
                        log_data = [
                            {'data': x, 'type': 'image', 'name': '%s/0_input' % self.stage, 'stage': self.stage},
                            {'data': gt, 'type': 'image', 'name': '%s/1_gt_mask' % self.stage, 'stage': self.stage},
                            {'data': pred, 'type': 'image', 'name': '%s/2_pred_mask' % self.stage, 'stage': self.stage}]
                        self.log_to_tensorboard(log_data)

                    time_left, it_time = timer()
                    print_in_place(
                        'Epoch: %d (validation) -- Batch: %d/%d -- ETA: %s -- It. time: %.4f s -- Mean loss: %f'
                        % (ep, i, self.cfg['val_steps_per_epoch'], time_left, it_time, ep_val_mean_loss.result()))

            self.log_to_tensorboard([{'data': ep_val_mean_loss.result(), 'type': 'scalar',
                                      'name': '%s/ep_mean_loss' % self.stage, 'stage': self.stage}],
                                    step=ep)

            print('')
            if ep_val_mean_loss.result() < self.state['best_val_loss']:
                print('\tMean validation loss decreased from %f to %f. Saving best model'
                      % (self.state['best_val_loss'], ep_val_mean_loss.result()))
                self.state['best_val_loss'] = ep_val_mean_loss.result()
                self.save_checkpoint('best_checkpoint')