import os
import json

import torch
import numpy as np


class TrainerBase:

    def __init__(self, trainer_mode, out_dir, batch_size, max_epochs, train_steps_per_epoch, val_steps_per_epoch, log_interval, **cfg_args):

        """
        Setup the trainer attributes
        :param out_dir: main output directory where logs and checkpoints will be saved
        :param batch_size:
        :param max_epochs: maximum training epoch number
        :param train_steps_per_epoch: train steps per epoch
        :param val_steps_per_epoch: validation steps per epoch
        :param log_interval: train/val steps between logs
        :param cfg_args: any additional configuration. It will be stored in self.cfg as a dict
        :return:
        """

        self.out_dir = out_dir
        self.log_dir = os.path.join(out_dir, 'logs')
        self.checkpoint_dir = os.path.join(out_dir, 'checkpoints')
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.train_steps_per_epoch = train_steps_per_epoch
        self.val_steps_per_epoch = val_steps_per_epoch
        self.log_interval = log_interval
        self.trainer_mode = trainer_mode

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.state = {'epoch': 0,
                      'train_step': 0,
                      'val_step': 0,
                      'best_val_loss': np.inf}

        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.cfg = cfg_args

    def setup(self):

        if self.trainer_mode == 'start':
            if len(os.listdir(self.checkpoint_dir)) > 0:
                raise Exception('Error! Output checkpoint dir (%s) not empty, which is incompatible with "%s" trainer mode'
                                % (self.checkpoint_dir, self.trainer_mode))
        elif self.trainer_mode == 'resume':
            if not len(os.listdir(self.checkpoint_dir)):
                raise Exception('Error! Cannot resume from an empty checkpoint dir. Use "start" trainer mode instead')
            self.load_last_checkpoint(self.checkpoint_dir)
        elif self.trainer_mode == 'test':
            if not len(os.listdir(self.checkpoint_dir)):
                raise Exception('Error! Cannot load best checkpoint from an empty checkpoint dir.')
            self.load_best_checkpoint(self.checkpoint_dir)
        elif self.trainer_mode == 'debug':
            pass
        else:
            raise Exception('Error! Input trainer mode (%s) not available' % self.trainer_mode)

    def set_model(self, model, device=None):
        self.model = model
        if device is not None:
            self.model.to(device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_fn(self, loss_fn, device=None):
        self.loss_fn = loss_fn
        if device is not None:
            self.loss_fn.to(device)

    def set_config(self, **cfg_args):
        self.cfg.update(cfg_args)

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

    def get_last_checkpoint(self):
        info_last_ckpt = os.path.join(self.checkpoint_dir, 'last_checkpoint.txt')
        try:
            with open(info_last_ckpt, 'r') as fp:
                last_checkpoint_name = fp.read().strip()
        except FileNotFoundError:
            print('"last_checkpoint.txt" not found in checkpoint directory: %s' % self.checkpoint_dir)
            last_checkpoint_name = None

        return last_checkpoint_name

    def save_checkpoint(self, name):
        """Saves a generic checkpoint dict to the output checkpoint directory. If working with custom checkpoints,
        this method must be overridden in the children class that inherits from TrainerBase

        :param name: output checkpoint filename without extension
        """

        # print('\tSaving checkpoint { %s } in %s' % (name, self.cfg['checkpoint_dir']))
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss_fn_state': self.loss_fn.state_dict(),
            'trainer_state': self.state,
            'trainer_cfg': self.cfg
        }

        ckpt_path = os.path.join(self.checkpoint_dir, name+'.pth')
        torch.save(checkpoint, ckpt_path)

        info_last_ckpt = os.path.join(self.checkpoint_dir, 'last_checkpoint.txt')
        with open(info_last_ckpt, 'w') as fp:
            fp.write(name + '.pth')

    def save_best_checkpoint(self):
        self.save_checkpoint('best_checkpoint')

    def load_checkpoint(self, ckpt_path):
        """Loads a generic checkpoint. If working with custom checkpoints, this method must be overridden
        in the children class that inherits from TrainerBase

        :param ckpt_path: path to the checkpoint file to be loaded
        """

        print('\tLoading checkpoint from: %s' % ckpt_path)
        checkpoint = torch.load(ckpt_path)

        self.state = checkpoint['trainer_state']
        self.loss_fn.load_state_dict(checkpoint['loss_fn_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.model.load_state_dict(checkpoint['model_state'])

    def load_last_checkpoint(self, ckpt_dir):

        print('\tLoading last checkpoint from folder: %s' % ckpt_dir)
        self.load_checkpoint(ckpt_path=os.path.join(ckpt_dir, self.get_last_checkpoint()))

    def load_best_checkpoint(self, ckpt_dir):
        print('\tLoading best checkpoint from folder: %s' % ckpt_dir)
        self.load_checkpoint(ckpt_path=os.path.join(ckpt_dir, 'best_checkpoint.pth'))
