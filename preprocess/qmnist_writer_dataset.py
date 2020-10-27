import torch

from torch.utils.data import Dataset


class WriterQMNIST(Dataset):
    """ QMNIST Dataset by Writer """

    def __init__(self, writer_id=0):
        self.writer_id = writer_id
        self._data = None
        self._labels = None
    
    def __len__(self):
        return self._data.shape[0]
    
    def __getitem__(self, idx):
        return self._data[idx], self._labels[idx]

    def add_datapoint(self, data, labels):
        if self._data is None:
            self._data = data.clone().detach()
            self._labels = labels.clone().detach()
        else:
            self._data = torch.cat((self._data, data), dim=0)
            self._labels = torch.cat((self._labels, labels), dim=0)
    
    @property
    def data(self):
        return self._data
    
    @property
    def labels(self):
        return self._labels
    


if __name__ == "__main__":
    dataset = WriterQMNIST()
    dataset.add_datapoint(torch.tensor([[1, 2, 3, 4]]), torch.tensor([0]))
    dataset.add_datapoint(torch.tensor([[1, 2, 3, 4]]), torch.tensor([0]))
    dataset.add_datapoint(torch.tensor([[1, 2, 3, 4]]), torch.tensor([1]))

    print(dataset.data)
    print(dataset.labels)