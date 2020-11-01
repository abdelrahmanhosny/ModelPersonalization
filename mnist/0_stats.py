import torch

from collections import defaultdict

from edgify.utils import WriterQMNIST

if __name__ == "__main__":
    train = defaultdict(lambda : 0)
    test = defaultdict(lambda : 0)

    train_size = 0
    test_size = 0
    for writer_id in range(131):
        ds_path = './data/QMNIST/train/w' + str(writer_id) + '.pth'
        train_ds = torch.load(ds_path)
        ds_path = './data/QMNIST/test/w' + str(writer_id) + '.pth'
        test_ds = torch.load(ds_path)
        print('{}, {},{},{}'.format(writer_id, len(train_ds), len(test_ds), len(train_ds) + len(test_ds)))
        train_size += len(train_ds)
        test_size += len(test_ds)
    print(train_size, test_size)
