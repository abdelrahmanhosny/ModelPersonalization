import os
import argparse
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Characterize training on the edge')

    # model parameters
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', \
        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--download', action='store_true', default=False,
                        help='Downloads dataset')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # setup 
    train_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': os.cpu_count(),
        'pin_memory': True
    }

    
    # data loading and transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10('data/CIFAR10', \
        download=args.download, transform=transform, train=True)
    train_dataloader = DataLoader(train_dataset, **train_kwargs)

    num_classes = 10
    model = torchvision.models.mobilenet_v2(pretrained=True)

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
    )

    model = model.to(device)
    
    # train
    model.train()
    optimizer = Adam(model.parameters(), lr=args.lr)

    for inputs, labels in train_dataloader:
        # forward pass
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = F.cross_entropy(output, labels)

        # backpropagation
        loss.backward()
        optimizer.step()
