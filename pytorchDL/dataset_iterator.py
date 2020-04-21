import torch


class DataIterator(object):

    def __init__(self, dataset, batch_size, num_workers, shuffle=True):
        """Core class for iterating over a pytorch dataset. This class handles proper shuffling application, batching and infinite
        iteration over the input dataset.

        :param dataset: a pytorch dataset object inheriting from pytorch Dataset class.
                        The dataset object must contain ".shuffle() method that shuffles data
        :param batch_size: integer that specifies the number of elements per batch
        :param num_workers: number of parallel dataloaders
        :param shuffle: boolean to specify if the data must be shuffled or not
        """

        self._dataset = dataset
        self._shuffle = shuffle
        if self._shuffle:
            self._dataset.shuffle()

        self._dataloader = torch.utils.data.DataLoader(self._dataset,
                                                       batch_size=batch_size,
                                                       shuffle=shuffle,
                                                       num_workers=num_workers)
        self.num_examples = len(self._dataset)
        self._dataset_iterator = iter(self._dataloader)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, item):
        return self._dataset[item]

    def __next__(self):
        try:
            batch_data = next(self._dataset_iterator)
        except StopIteration:  # when dataset is exhausted, re shuffle (if applicable) it and reload iterator
            if self._shuffle:
                self._dataset.shuffle()
            self._dataset_iterator = iter(self._dataloader)
            batch_data = next(self._dataset_iterator)

        return batch_data
