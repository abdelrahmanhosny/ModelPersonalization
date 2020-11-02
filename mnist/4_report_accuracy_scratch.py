import os
import torch

from collections import defaultdict

from edgify.utils import WriterQMNIST

if __name__ == "__main__":
    for writer_id in range(131):
        with open(os.path.join('data', 'QMNIST', 'models-subject-out', str(writer_id), 'accuracy_scratch.txt'), 'r') as f:
            accuracy_scratch = f.readline().strip()
        print(accuracy_scratch)
        