import argparse
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib import cm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from edgify.utils import WriterQMNIST
from edgify.models import MNISTClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test accuracy of the general MNIST model per writer')
    parser.add_argument('--data_dir', type=str, default='data/QMNIST/test',
                        help='Directory where the per-writer test dataset was created')
    parser.add_argument('--model', type=str, default='data/mnist_classifier-50.pt',
                        help='PyTorch model to test')
    args = parser.parse_args()

    writers = []
    accuracies = []

    writers_datasets = [f for f in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, f)) and f.endswith(".pth")]
    
    for writer_dataset in writers_datasets:
        dataset = torch.load(os.path.join(args.data_dir, writer_dataset))
        dataloader = DataLoader(dataset, batch_size=64)

        
        model = MNISTClassifier()
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
        model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                data = data.reshape((-1, 1, 28, 28))
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(dataloader.dataset)

        print('Writer {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            writer_dataset.split('.')[0], test_loss, correct, len(dataloader.dataset),
            100. * correct / len(dataloader.dataset)
        ))

        writers.append(writer_dataset.split('.')[0])
        accuracies.append(100. * correct / len(dataloader.dataset))
    
    print('Minimum: {:.0f}%'.format(min(accuracies)))
    print('Maximum: {:.0f}%'.format(max(accuracies)))
    print('Average: {:.0f}%'.format(sum(accuracies) / len(accuracies)))
    
    with open('data/accuracies.csv', 'w') as f:
        for i in range(len(writers)):
            f.write('{}, {}\n'.format(writers[i], accuracies[i]))

    colors = cm.get_cmap('viridis', 4)
    plt.hist(accuracies, color=colors.colors[1], edgecolor='black')
    plt.ylabel('Writers')
    plt.xlabel('Accuracy (%)')
    plt.title('Epochs: {}. Avg. Accuracy: {:.2f}%'.format(50, sum(accuracies) / len(accuracies)))
    plt.show()
    

