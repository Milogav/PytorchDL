import os
import json

from .utils.misc import get_current_time

import tensorflow as tf
from tensorboard import program
import torch
import numpy as np


class TrainerBase:

    def __init__(self, out_dir, batch_size, max_epochs, train_steps_per_epoch, val_steps_per_epoch, log_interval, **kwargs):

        """
        Setup the trainer configuration
        :param out_dir: main output directory where logs and checkpoints will be saved
        :param batch_size:
        :param max_epochs: maximum training epoch number
        :param train_steps_per_epoch: train steps per epoch
        :param val_steps_per_epoch: validation steps per epoch
        :param log_interval: train/val steps between logs
        :param kwargs: any additional configuration arguments
        :return:
        """

        self.cfg = {'out_dir': out_dir,
                    'log_dir': os.path.join(out_dir, 'logs'),
                    'checkpoint_dir': os.path.join(out_dir, 'checkpoints'),
                    'batch_size': batch_size,
                    'max_epochs': max_epochs,
                    'train_steps_per_epoch': train_steps_per_epoch,
                    'val_steps_per_epoch': val_steps_per_epoch,
                    'log_interval': log_interval}

        os.makedirs(self.cfg['log_dir'], exist_ok=True)
        os.makedirs(self.cfg['checkpoint_dir'], exist_ok=True)

        self.cfg.update(kwargs)  # append extra fields in cfg dictionary
        self.state = {'epoch': 0,
                      'train_step': 0,
                      'val_step': 0,
                      'best_val_loss': np.inf}

        self.extra = {}
        self.summary_writer = None
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def save_config(self, out_path):
        """
        Export the trainer configuration to a json file
        """

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as fp:
            json.dump(self.cfg, fp)

    def load_config(self, cfg_json):
        """
        Import a previously exported trainer configuration from a json file
        """

        with open(cfg_json, 'r') as fp:
            self.cfg = json.load(fp)

    def create_tensorboard_summary(self, launch_tensorboard=True):
        """
        Initialize a tensorboard summary in {out_dir}/logs/{current_time}
        """

        self.cfg['summary_dir'] = os.path.join(self.cfg['log_dir'], get_current_time())
        self.summary_writer = tf.summary.create_file_writer(self.cfg['summary_dir'], flush_millis=10000)
        if launch_tensorboard:
            self.launch_tensorboard()

    def launch_tensorboard(self, port=3468):
        """
        Launch tensorboard in background to locally inspect the created summary under a specified port
        """
        self._tb = program.TensorBoard()
        self._tb.configure(argv=[None, '--logdir', self.cfg['summary_dir'], '--port', str(port)])
        self.extra['tensorboard_url'] = self._tb.launch()
        self.print_tensorboard_url()

    def print_tensorboard_url(self):
        print('\n\nTensorboard url: %s\n\n' % self.extra.get('tensorboard_url', 'NOT_CREATED'))

    def log_to_tensorboard(self, log_data, max_images=1):
        """
        Log a list of data to tensorboard. Step is automatically determined from the trainer current state.
        Each piece of data to be logged must be defined as a dict, with fields:
            'type': type of data ('scalar' or 'image')
            'name': the name of the log in which this new data will be included
            'stage': 'train', 'val', 'test'
            'data': numpy array or tensor representing the data

        :param log_data: list of dicts, each one containing a data to be log. This dict must have 'type', 'name', 'stage' and 'data' fields
        :param max_images: maximum number of images to output at each log step
        """

        with self.summary_writer.as_default():
            for data_dict in log_data:
                step = self.state[data_dict['stage'].lower() + '_step']
                if data_dict['type'] == 'scalar':
                    tf.summary.scalar(data_dict['name'], data_dict['data'], step=step)
                elif data_dict['type'] == 'image':
                    tf.summary.image(data_dict['name'],  data_dict['data'], step=step, max_outputs=max_images)

    def get_last_checkpoint(self):
        info_last_ckpt = os.path.join(self.cfg['checkpoint_dir'], 'last_checkpoint.txt')
        try:
            with open(info_last_ckpt, 'r') as fp:
                last_checkpoint_name = fp.read()
        except FileNotFoundError:
            print('"last_checkpoint.txt" not found in checkpoint directory: %s' % self.cfg['checkpoint_dir'])
            last_checkpoint_name = None

        return last_checkpoint_name

    def save_checkpoint(self, name):

        print('\tSaving checkpoint { %s } in %s' % (name, self.cfg['checkpoint_dir']))
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss_fn_state': self.loss_fn.state_dict(),
            'trainer_cfg': self.cfg,
            'trainer_state': self.state
        }

        ckpt_path = os.path.join(self.cfg['checkpoint_dir'], name+'.pth')
        torch.save(checkpoint, ckpt_path)

        info_last_ckpt = os.path.join(self.cfg['checkpoint_dir'], 'last_checkpoint.txt')
        with open(info_last_ckpt, 'w') as fp:
            fp.write(name + '.pth')

    def load_checkpoint(self, ckpt_path):

        print('\tLoading checkpoint from: %s' % ckpt_path)
        checkpoint = torch.load(ckpt_path)

        self.cfg = checkpoint['trainer_cfg']
        self.state = checkpoint['trainer_state']
        self.loss_fn.load_state_dict(checkpoint['loss_fn_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.model.load_state_dict(checkpoint['model_state'])

    def load_last_checkpoint(self, ckpt_dir):

        print('\tLoading last checkpoint from folder: %s' % ckpt_dir)
        self.cfg['checkpoint_dir'] = ckpt_dir
        self.load_checkpoint(ckpt_path=os.path.join(ckpt_dir, self.get_last_checkpoint()))
