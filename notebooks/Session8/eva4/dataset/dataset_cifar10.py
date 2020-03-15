import torch
from torchvision import datasets, transforms

from .dataset_base import BaseDataset

class CIFAR10Dataset(BaseDataset):
    KEY = "CIFAR10"
    CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, dataset_config, train_transforms, test_transforms):
        self.config = dataset_config

        if train_transforms is None:
            train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.config.dataset_mean, self.config.dataset_mean),
                ])
        self.train_transforms = train_transforms

        if test_transforms is None:
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.config.dataset_mean, self.config.dataset_mean),
                ])
        self.test_transforms = test_transforms

        self.reload()

    def reload(self):
        self.train_set = datasets.CIFAR10(root=self.config.dataset_root, train=True, download=True, transform=self.train_transforms)
        self.test_set = datasets.CIFAR10(root=self.config.dataset_root, train=True, download=True, transform=self.test_transforms)

        dataloader_args = dict(
            shuffle=self.config.shuffle,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        
        self.train_loader = torch.utils.data.DataLoader(self.train_set, **dataloader_args)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, **dataloader_args)

