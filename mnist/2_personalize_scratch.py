import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
from edgify.models import MNISTClassifier
from edgify.utils import WriterQMNIST


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train on Edge')

    # model parameters
    parser.add_argument('--data_dir', type=str, default='data/QMNIST',
                        help='Directory where the per-writer test dataset was created')
    parser.add_argument('--writer_id', type=int, default=0, metavar='N', \
        help='writer ID to leave out of general training (default: 0)')
    parser.add_argument('--model', type=str, default='data/mnist_classifier-10.pt',
                        help='PyTorch model to test')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', \
        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
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
    test_kwargs = {
        'batch_size': args.test_batch_size
    }
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    for writer_id in range(131):
        print('Building dataset for writer ', writer_id)
        train_dataset = torch.load(os.path.join(args.data_dir, 'train', 'w' + str(writer_id) + '.pth' ))
        test_dataset = torch.load(os.path.join(args.data_dir, 'test', 'w' + str(writer_id) + '.pth' ))

        train_dataloader = DataLoader(train_dataset, **train_kwargs)
        test_dataloader = DataLoader(test_dataset, **test_kwargs)

        print('Loaded a training dataset of size: ', len(train_dataset))
        print('Loaded a writer test dataset of size: ', len(test_dataset))

        model = MNISTClassifier().to(device)

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

        # test on writer
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                data = data.reshape((-1, 1, 28, 28))
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_dataloader.dataset)

        print('Writer {}: Accuracy: {}/{} ({:.2f}%)'.format(
            writer_id, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)
        ))

        # logging and saving
        save_dir = os.path.join(args.data_dir, 'models-subject-out', str(writer_id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.save(model.state_dict(), os.path.join(save_dir, 'scratch-model.pt'))

        with open(os.path.join(save_dir, 'accuracy_scratch.txt'), 'w') as f:
            f.write('{:.2f}\n'.format(100. * correct / len(test_dataloader.dataset)))
