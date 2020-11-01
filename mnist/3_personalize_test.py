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
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
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
    test_kwargs = {
        'batch_size': args.test_batch_size
    }
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        test_kwargs.update(cuda_kwargs)

    test_dataset = torch.load(os.path.join(args.data_dir, 'test', 'w' + str(args.writer_id) + '.pth' ))
    test_dataloader = DataLoader(test_dataset, **test_kwargs)

    model = MNISTClassifier()
    model_path = os.path.join(args.data_dir, 'models-subject-out', str(args.writer_id), 'personalized-model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))

    # test on writer
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.reshape((-1, 1, 28, 28))
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset)

    print('Writer {}: Accuracy: {}/{} ({:.2f}%)'.format(
        args.writer_id, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)
    ))

    # logging and saving
    save_dir = os.path.join(args.data_dir, 'models-subject-out', str(args.writer_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'accuracy_after.txt'), 'w') as f:
        f.write('{:.2f}\n'.format(100. * correct / len(test_dataloader.dataset)))
