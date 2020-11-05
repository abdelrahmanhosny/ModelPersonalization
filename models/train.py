import os
import argparse
import cProfile, pstats
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

from datetime import datetime
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import Adam
from torchvision import datasets, transforms



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Characterize training on the edge')

    # model parameters
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', \
        help='input batch size for training (default: 64)')
    parser.add_argument('--num_batches', type=int, default=100, metavar='N', \
        help='number of batches to train for (not a full epoch)')
    parser.add_argument('--output_dir', type=str, required=True, \
        help='output directory')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--freeze', action='store_true', default=False,
                        help='freezes feature extraction layers')
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
        'pin_memory': True,
        'sampler': SubsetRandomSampler(range(args.batch_size * args.num_batches))
    }

    
    # data loading and transform
    started = datetime.now()
    print('Data Loading @ ', datetime.now())
    pr = cProfile.Profile()
    pr.enable()
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

    if args.freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
    )

    model = model.to(device)

    pr.disable()
    with open(os.path.join(args.output_dir, str(args.batch_size) + '_dataloading.prof'), 'w') as f:
        ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
        ps.print_stats()
    pr.clear()

    # train
    print('Training @ ', datetime.now())
    model.train()
    optimizer = Adam(model.parameters(), lr=args.lr)

    forward_profiler = cProfile.Profile()
    backward_profiler = cProfile.Profile()

    for batch_index, (inputs, labels) in enumerate(train_dataloader):
        # forward pass
        forward_profiler.enable()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = F.cross_entropy(output, labels)
        forward_profiler.disable()

        # backpropagation
        backward_profiler.enable()
        loss.backward()
        optimizer.step()
        backward_profiler.disable()

        print(batch_index)
        if (batch_index + 1) % args.num_batches == 0:
            break
    
    with open(os.path.join(args.output_dir, str(args.batch_size) + '_train_forward.prof'), 'w') as f:
        ps = pstats.Stats(forward_profiler, stream=f).sort_stats('cumulative')
        ps.print_stats()
    with open(os.path.join(args.output_dir, str(args.batch_size) + '_train_backprob.prof'), 'w') as f:
        ps = pstats.Stats(backward_profiler, stream=f).sort_stats('cumulative')
        ps.print_stats()

    finished = datetime.now()
    print('Finished @ ', finished, ' after ', finished - started)