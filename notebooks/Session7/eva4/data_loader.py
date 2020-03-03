import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


from eva4.common import reset_seed, get_config, get_device

def get_train_data(train_transforms, config):
  if train_transforms is None:
    train_transforms = transforms.Compose([transforms.ToTensor()])
  
  train = None
  if config.get("dataset") == "MNIST":
    train = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)

  if config.get("dataset") == "CIFAR10":
    train = datasets.CIFAR10('../data', train=True, download=True, transform=train_transforms)

  return train

def get_train_loader(train_transforms, config):

  train = get_train_data(train_transforms, config)
  
  shuffle = config.get("shuffle")
  batch_size = config.get("batch_size")
  num_workers = config.get("num_workers")
  pin_memory = config.get("pin_memory")

  # dataloader arguments - something you'll fetch these from cmdprmt
  dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

  # train dataloader
  train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
  return train_loader

def print_train_data_stats(train, train_loader):
  train_data = train.train_data
  train_data = train.transform(train_data.numpy())

  print('[Train]')
  print(' - Numpy Shape:', train.train_data.cpu().numpy().shape)
  print(' - Tensor Shape:', train.train_data.size())
  print(' - min:', torch.min(train_data))
  print(' - max:', torch.max(train_data))
  print(' - mean:', torch.mean(train_data))
  print(' - std:', torch.std(train_data))
  print(' - var:', torch.var(train_data))

  dataiter = iter(train_loader)
  images, labels = dataiter.next()

  print(images.shape)
  print(labels.shape)

  # Let's visualize some of the images
  plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

  figure = plt.figure()
  num_of_images = 60
  for index in range(1, num_of_images + 1):
      plt.subplot(6, 10, index)
      plt.axis('off')
      plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

def get_test_data(test_transforms, config):
  if test_transforms is None:
    test_transforms = transforms.Compose([transforms.ToTensor()])

  test = None
  if config.get("dataset") == "MNIST":
    test = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

  if config.get("dataset") == "CIFAR10":
    test = datasets.CIFAR10('../data', train=False, download=True, transform=test_transforms)

  return test

def get_test_loader(test_transforms, config):

  test = get_test_data(test_transforms, config)

  shuffle = config.get("shuffle")
  batch_size = config.get("batch_size")
  num_workers = config.get("num_workers")
  pin_memory = config.get("pin_memory")

  # dataloader arguments - something you'll fetch these from cmdprmt
  dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

  # test dataloader
  test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
  return test_loader

def print_test_data_stats(test, test_loader):
  test_data = test.test_data
  test_data = test.transform(test_data.numpy())

  print('[Test]')
  print(' - Numpy Shape:', test.test_data.cpu().numpy().shape)
  print(' - Tensor Shape:', test.test_data.size())
  print(' - min:', torch.min(test_data))
  print(' - max:', torch.max(test_data))
  print(' - mean:', torch.mean(test_data))
  print(' - std:', torch.std(test_data))
  print(' - var:', torch.var(test_data))

  dataiter = iter(test_loader)
  images, labels = dataiter.next()

  print(images.shape)
  print(labels.shape)

  # Let's visualize some of the images
  plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

  figure = plt.figure()
  num_of_images = 60
  for index in range(1, num_of_images + 1):
      plt.subplot(6, 10, index)
      plt.axis('off')
      plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')