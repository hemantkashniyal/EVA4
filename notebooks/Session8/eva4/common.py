import torch
import random 
import numpy as np
from environs import Env
import matplotlib.pyplot as plt

def cuda_available():
  return torch.cuda.is_available()

def reset_seed(seed_value=1):
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)

  if cuda_available():
      torch.cuda.manual_seed(seed_value)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def get_mean_std(data_type, exp):
  exp_data = exp.data.numpy()
  exp_data = exp.transform(exp_data)

  print('[{}]'.format(data_type))
  print(' - Numpy Shape:', exp.data.cpu().numpy().shape)
  print(' - Tensor Shape:', exp.data.size())
  print(' - min:', torch.min(exp_data))
  print(' - max:', torch.max(exp_data))
  print(' - mean:', torch.mean(exp_data))
  print(' - std:', torch.std(exp_data))
  print(' - var:', torch.var(exp_data))

  data_mean = torch.mean(exp_data).item()
  print("{} data_mean: {}".format(data_type, data_mean))

  data_std = torch.std(exp_data).item()
  print("{} data_std: {}".format(data_type, data_std))

  return data_mean, data_std

def print_stats(data, data_loader):
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
  import matplotlib.pyplot as plt

  plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

def get_config(env):

  input_size = tuple(env.list("APP_INPUT_SIZE", subcast=int))
  input_channel = input_size[0]
  input_dimension = input_size[1:]

  batch_size = env.int("APP_DATA_LOADER_BATCH_SIZE")
  num_workers = 4
  pin_memory = True
  if not cuda_available():
    batch_size = 64
    num_workers = 1
    pin_memory = True

  config = {
      # experiment config
      "cuda_available": cuda_available(),
      "device": "cuda" if cuda_available() else "cpu",
      "epochs": env.int("APP_EPOCHS"),
      "desired_accuracy": env.float("APP_DESIRED_ACCURACY"),
      "break_on_reaching_desired_accuracy": env.bool("APP_BREAK_ON_REACHING_DESIRED_ACCURACY"),
      "consecutive_desired_accuracy": env.int("APP_CONSECUTIVE_DESIRED_ACCURACY"),
      "input_size": input_size,
      "input_channel": input_channel,
      "input_dimension": input_dimension,
      
      # data_loader config
      "dataset": env("APP_DATASET"),
      "batch_size": batch_size,
      "shuffle": env.bool("APP_DATA_LOADER_SHUFFLE"),
      "num_workers": num_workers,
      "pin_memory": pin_memory,

      # network config
      "dropout": env.float("APP_DROPOUT"),
      "bias_enabled": env.bool("APP_BIAS_ENABLED"),

      # optimizer config
      "optimizer": env("APP_OPTIMIZER"),
      "learning_rate": env.float("APP_OPTIMIZER_LEARNING_RATE"),
      "momentum": env.float("APP_OPTIMIZER_MOMENTUM"),

      # scheduler config
      "scheduler": env("APP_SCHEDULER"),
      "step_size": env.int("APP_SCHEDULER_STEP_SIZE"),
      "milestones": env.list("APP_SCHEDULER_MILESTONES", subcast=int),
      "gamma": env.float("APP_SCHEDULER_GAMMA"),
  }
  return config

def get_device(config):
  device_str = config.get("device")
  device = torch.device(device_str)
  return device