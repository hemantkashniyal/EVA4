from .dataset_cifar10 import CIFAR10Dataset
from .dataset_base import BaseDataset

class DatasetManager(object):
    datasets = {
        CIFAR10Dataset.KEY: CIFAR10Dataset
    }

    @classmethod
    def get_dataset(cls, dataset_config, train_transforms, test_transforms) -> BaseDataset:
        dataset = DatasetManager.datasets.get(dataset_config.type)(dataset_config, train_transforms, test_transforms)
        return dataset

