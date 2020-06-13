from time import time
from datetime import datetime

import numpy as np


def print_in_place(text):
    print('\r' + text + '\t' * 5, end='', sep='')


class Timer:

    def __init__(self, total_steps):
        self.start_time = time()
        self.total_steps = total_steps
        self.it = 0

    def __call__(self):
        self.it += 1

        elapsed_time = time() - self.start_time
        est_time_per_step = elapsed_time / self.it
        time_left = est_time_per_step * (self.total_steps - self.it)
        m, s = divmod(time_left, 60)
        h, m = divmod(m, 60)
        time_left = "%d:%02d:%02d" % (h, m, s)
        return time_left, est_time_per_step


def get_current_time():
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


def generate_random_colors(n_colors, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    colors = np.random.uniform(0, 1, size=(n_colors, 3))
    if random_seed is not None:
        np.random.seed()

    return colors
