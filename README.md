# Model Personalization on Edge

Improving the performance of DL models for individual users by re-training on a user's data on the edge.


## Setup

```Python
python setup.py install
```

## QMNIST Dataset

Divide the datset by the writer ID.

```Python
python filter_by_user.py --download --dataset=train
python filter_by_user.py --download --dataset=test
```

The resulting user-specific datasets will be under `data/QMNIST/train` and `data/QMNIST/test`. 
The file naming convention is `w-<witer_id>.pth`.

To load the dataset of a specific writer:

```Python
import torch
from torch.utils.data import DataLoader

dataset = torch.load('data/QMNIST/train/w-<writer_id>.pth')
dataloader = DataLoader(dataset)
```


## License
BSD 3-Clause License.