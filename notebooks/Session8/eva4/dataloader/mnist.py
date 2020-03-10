import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


from eva4.common import reset_seed, get_config, get_device

DATA_CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def get_mean_std():
    data_mean = (0.1296318918466568,)
    print("{} data_mean: {}".format("MNIST", data_mean))

    data_std = (0.12316563725471497)
    print("{} data_std: {}".format("MNIST", data_std))

    return data_mean, data_std


def get_data_loader(train_transforms, test_transforms, config):
    if train_transforms is None:
        train_transforms = transforms.Compose([transforms.ToTensor()])
    
    if test_transforms is None:
        test_transforms = transforms.Compose([transforms.ToTensor()])
  
    train = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)

    test = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

    shuffle = config.get("shuffle")
    batch_size = config.get("batch_size")
    num_workers = config.get("num_workers")
    pin_memory = config.get("pin_memory")

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return train_loader, test_loader

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def print_data_stats(loader, config, images=10):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    