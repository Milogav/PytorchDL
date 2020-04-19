import os
import json

from .utils.misc import get_current_time

import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

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
        self.summary_writer = SummaryWriter(self.cfg['summary_dir'], flush_secs=15)
        if launch_tensorboard:
            self.launch_tensorboard()

    def launch_tensorboard(self, port=3468):
        """
        Launch tensorboard in background to locally inspect the created summary under a specified port
        """
        self._tb = program.TensorBoard()
        self._tb.configure(argv=[None, '--logdir', os.path.dirname(self.cfg['summary_dir']), '--port', str(port)])
        self.extra['tensorboard_url'] = self._tb.launch()
        self.print_tensorboard_url()

    def print_tensorboard_url(self):
        print('\n\nTensorboard url: %s\n\n' % self.extra.get('tensorboard_url', 'NOT_CREATED'))

    def log_to_tensorboard(self, log_data, step=None):
        """
        Log a list of data to tensorboard. Step is automatically determined from the trainer current state.
        Each piece of data to be logged must be defined as a dict, with fields:
            'type': type of data ('scalar' or 'image')
            'name': the name of the log in which this new data will be included
            'stage': 'train', 'val', 'test'
            'data': numpy array or torch tensor representing the data.
                    If image data, use NCHW format, float type and intensity range between [0, 1]
                    If pointcloud data, use Nx6xP format, where P is the number of points and the first dimension represents [x y z R G B]
                        RGB values must be in the range [0, 1]

        :param log_data: list of dicts, each one containing a data to be log. This dict must have 'type', 'name', 'stage' and 'data' fields
        """

        for data_dict in log_data:
            if step is None:
                step = self.state[data_dict['stage'].lower() + '_step']

            if data_dict['type'] == 'scalar':
                self.summary_writer.add_scalar(tag=data_dict['name'], scalar_value=data_dict['data'], global_step=step)
            elif data_dict['type'] == 'image':
                self.summary_writer.add_images(tag=data_dict['name'], img_tensor=data_dict['data'], global_step=step)
            elif data_dict['type'] == 'pointcloud':
                vertices = data_dict['data'][:, 0:3, :].permute(0, 2, 1)

                colors = 255 * data_dict['data'][:, 3:6, :].permute(0, 2, 1)
                colors = colors.type(torch.uint8)
                self.summary_writer.add_mesh(tag=data_dict['name'], vertices=vertices, colors=colors, global_step=step)

    def get_last_checkpoint(self):
        info_last_ckpt = os.path.join(self.cfg['checkpoint_dir'], 'last_checkpoint.txt')
        try:
            with open(info_last_ckpt, 'r') as fp:
                last_checkpoint_name = fp.read().strip()
        except FileNotFoundError:
            print('"last_checkpoint.txt" not found in checkpoint directory: %s' % self.cfg['checkpoint_dir'])
            last_checkpoint_name = None

        return last_checkpoint_name

    def save_checkpoint(self, name):
        """Saves a generic checkpoint dict to the output checkpoint directory. If working with custom checkpoints,
        this method must be overridden in the children class that inherits from TrainerBase

        :param name: output checkpoint filename without extension
        """

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
        """Loads a generic checkpoint. If working with custom checkpoints, this method must be overridden
        in the children class that inherits from TrainerBase

        :param ckpt_path: path to the checkpoint file to be loaded
        """

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
