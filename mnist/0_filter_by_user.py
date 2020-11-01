import argparse
import torch
import os

from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from edgify.utils import WriterQMNIST


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter QMNIST dataset by writer')
    parser.add_argument('--dataset', type=str, default='train',
                        help='Choose to filter the train or the test datasets')
    parser.add_argument('--download', action='store_true', default=False,
                        help='Download the dataset if not already downloaded')
    
    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])

    qmnist_dataset = datasets.QMNIST('./data', what=args.dataset, download=args.download, compat=False, transform=transform)

    dataset = defaultdict(lambda: WriterQMNIST())

    for sample in qmnist_dataset:
        data, label = sample
        writer_id = label[3]
        class_id = label[0]
        dataset[writer_id.item()].add_datapoint(data, torch.tensor([class_id]))
            
    save_dir = os.path.join('data', 'QMNIST', args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for writer_id, writer_dataset in dataset.items():
        torch.save(writer_dataset, os.path.join(save_dir, 'w' + str(writer_id) + '.pth'))
    