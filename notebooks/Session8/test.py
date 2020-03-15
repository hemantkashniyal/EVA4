import sys,os
sys.path.append(os.getcwd())


import json
from environs import Env

from eva4.config.config import TrainingConfig
env = Env()
env.read_env("./cifar10.experiment.env.txt", recurse=False)

config = TrainingConfig()
print("Experiment Config: ", json.dumps(config, default=lambda x: x.__dict__, sort_keys=False, indent=4))

from imageaug.transforms import Colorspace, RandomAdjustment, RandomRotatedCrop
input_size = config.dataset.input_size
crop_size = config.dataset.input_dimension
angle_std = 7 # in degrees

dataset_std = config.dataset.dataset_std
dataset_mean = config.dataset.dataset_mean

from torchvision import transforms
# Define training transforms
train_transforms = transforms.Compose([
    transforms.RandomCrop(config.dataset.input_dimension, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(dataset_mean, dataset_std),
])
# train_transforms = None

# Define test transforms
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_mean, dataset_std),
])
# test_transforms = None

from eva4.dataset.dataset_manager import DatasetManager
TrainingConfig.print(config.dataset)
dataset = DatasetManager.get_dataset(config.dataset, train_transforms, test_transforms)

from eva4.network.network_manager import NetworkManager
TrainingConfig.print(config.network)
# from eva4.network.cifar10.cifar10 import CIFAR10Net
# network = CIFAR10Net(config.network)
from eva4.network.resnet.resnet import ResNet18
network = ResNet18()

network = network.to(config.device)
NetworkManager.summarize(network, config.network, config.dataset)

from eva4.loss_fn.loss_fn_manager import LossFnManager
TrainingConfig.print(config.loss_fn)
loss = LossFnManager.get_loss_fn(config.loss_fn)

from eva4.optimizer.optimizer_manager import OptimizerManager
TrainingConfig.print(config.optimizer)
optimizer = OptimizerManager.get_optimizer(network, config.optimizer)

from eva4.scheduler.scheduler_manager import SchedulerManager
TrainingConfig.print(config.scheduler)
scheduler = SchedulerManager.get_scheduler(optimizer, config.scheduler)

from eva4.regularizer.regularizer_manager import RegularizerManager
TrainingConfig.print(config.regularizer)
regularizer = RegularizerManager.get_regularizer(config.regularizer)

from eva4.training.training_manager import TrainingManage
training = TrainingManager(config, dataset, network, loss, scheduler, optimizer, regularizer)
training.start()
training.summarize()

