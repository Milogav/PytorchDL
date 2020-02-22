from time import time
from datetime import datetime


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


