import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
from edgify.models import MNISTClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train on Edge')

    # model parameters
    parser.add_argument('--data_dir', type=str, default='data/QMNIST',
                        help='Directory where the per-writer test dataset was created')
    parser.add_argument('--writer_id', type=int, default=0, metavar='N', \
        help='writer ID to personalize model for (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', \
        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # setup 
    train_kwargs = {
        'batch_size': args.batch_size
    }
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)

    train_dataset = torch.load(os.path.join(args.data_dir, 'train', 'w' + str(args.writer_id) + '.pth' ))
    train_dataloader = DataLoader(train_dataset, **train_kwargs)

    model = MNISTClassifier().to(device)
    model_path = os.path.join(args.data_dir, 'models-subject-out', str(args.writer_id), 'general-model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))

    # train
    model.train()
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):    
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data = data.reshape((-1, 1, 28, 28))
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()
                ))

    # logging
    save_dir = os.path.join(args.data_dir, 'models-subject-out', str(args.writer_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(model.state_dict(), os.path.join(save_dir, 'personalized-model.pt'))
