import os

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from pytorchDL.utils.misc import get_current_time
from tensorboard import program
from tqdm import trange


class TensorboardLogger:

    def __init__(self, log_dir, port=3468):
        self.summary_dir = os.path.join(log_dir, get_current_time())
        self.summary_writer = SummaryWriter(self.summary_dir, flush_secs=15)
        self._launch_tensorboard(port=port)

    def _launch_tensorboard(self, port):
        """
        Launch tensorboard in background to locally inspect the created summary under a specified port
        """
        self._tb = program.TensorBoard()
        self._tb.configure(argv=[None, '--logdir', os.path.dirname(self.summary_dir), '--port', str(port)])
        self.tensorboard_url = self._tb.launch()
        self.print_tensorboard_url()

    def print_tensorboard_url(self):
        print('\n\nTensorboard url: %s\n\n' % self.tensorboard_url)

    def log(self, log_data, stage, step):
        """
        Log a list of data to tensorboard. Step is automatically determined from the trainer current state.
        Each piece of data to be logged must be defined as a dict, with fields:
            'type': type of data ('scalar' or 'image')
            'name': the name of the log in which this new data will be included (e.g 'batch_loss', 'accuracy')
            'data': numpy array or torch tensor representing the data.
                    If image data, use NCHW format, float type and intensity range between [0, 1]
                    If pointcloud data, use Nx6xP format, where P is the number of points and the first dimension represents [x y z R G B]
                        RGB values must be in the range [0, 1]

        :param log_data: list of dicts, each one containing a piece of data to be logged. This dict must have 'type', 'name' and 'data' fields
        :param stage: string naming the training stage for the data to log. (e.g 'train', 'val', 'test')
        """

        for data_dict in log_data:

            tag = '%s/%s' % (stage, data_dict['name'])
            if data_dict['type'] == 'scalar':
                self.summary_writer.add_scalar(tag=tag, scalar_value=data_dict['data'], global_step=step)
            elif data_dict['type'] == 'image':
                self.summary_writer.add_images(tag=tag, img_tensor=data_dict['data'], global_step=step)
            elif data_dict['type'] == 'pointcloud':
                if isinstance(data_dict['data'], np.ndarray):
                    data_dict['data'] = torch.from_numpy(data_dict['data'])
                vertices = data_dict['data'][:, 0:3, :].permute(0, 2, 1)

                colors = 255 * data_dict['data'][:, 3:6, :].permute(0, 2, 1)
                colors = colors.type(torch.uint8)
                self.summary_writer.add_mesh(tag=tag, vertices=vertices, colors=colors, global_step=step)
            else:
                raise Exception('Logging input data type (%s) is not implemented' % data_dict['type'])


class ProgressLogger:

    def __init__(self, total_steps, description='Progress'):
        self.prog_bar = trange(total_steps, desc=description)

    def log(self, **kwargs):
        self.prog_bar.set_postfix(kwargs)
        self.prog_bar.update()

    def prepend_text(self, text):
        self.prog_bar.write(text)

    def close(self):
        self.prog_bar.close()
