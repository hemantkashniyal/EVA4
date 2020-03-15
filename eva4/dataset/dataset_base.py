import torch
from torchvision import datasets, transforms

class BaseDataset(object):
    def reload(self):
        raise NotImplementedError