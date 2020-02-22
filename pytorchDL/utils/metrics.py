
class MeanMetric:

    def __init__(self):
        self.acc = 0
        self.it = 0

    def __call__(self, value):

        self.acc += value
        self.it += 1

    def result(self):
        return self.acc / self.it
