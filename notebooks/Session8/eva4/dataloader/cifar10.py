import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


from eva4.common import reset_seed, get_config, get_device

DATA_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_mean_std():
    data_mean = (0.4914, 0.4822, 0.4465)    
    print("{} data_mean: {}".format("CIFAR10", data_mean))

    data_std = (0.2470, 0.2435, 0.2616)
    print("{} data_std: {}".format("CIFAR10", data_std))

    return data_mean, data_std


def get_data_loader(train_transforms, test_transforms, config):
    if train_transforms is None:
        train_transforms = transforms.Compose([transforms.ToTensor()])
    
    if test_transforms is None:
        test_transforms = transforms.Compose([transforms.ToTensor()])
  
    train = datasets.CIFAR10('../data', train=True, download=True, transform=train_transforms)

    test = datasets.CIFAR10('../data', train=False, download=True, transform=test_transforms)

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

def imshow(image):
    image = image / 2 + 0.5
    np_image = image.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))

def print_data_stats(loader, config, images=20):
    device = config.get("device")
    count  = 0
    figure = plt.figure(figsize=(10,9))
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        for index, label in enumerate(target):
          title = "{}".format(DATA_CLASSES[label.item()])
          axis = figure.add_subplot(4, 5, count+1, xticks=[], yticks=[])
          axis.set_title(title)
          imshow(data[index].cpu())
          
          count += 1  
          if count == images:
            break
        if count == images:
            break
    return