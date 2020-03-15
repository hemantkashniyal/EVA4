import sys,os
sys.path.append(os.getcwd())


import json
from torchsummary import summary
from environs import Env

from torchvision import datasets, transforms
from eva4.config.config import TrainingConfig
env = Env()
env.read_env("./cifar10.experiment.env.txt", recurse=False)

config = TrainingConfig()
print("Experiment Config: ", json.dumps(config, default=lambda x: x.__dict__, sort_keys=False, indent=4))


train_transforms = None
test_transforms = None

from imageaug.transforms import Colorspace, RandomAdjustment, RandomRotatedCrop

input_size = config.dataset.input_size
crop_size = config.dataset.input_dimension
angle_std = 7 # in degrees


# Define training transforms
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_transforms = None

# Define test transforms
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_transforms = None

from eva4.dataset.dataset_manager import DatasetManager

dataset = DatasetManager.get_dataset(config.dataset, train_transforms, test_transforms)

from eva4.network.network_manager import NetworkManager

from eva4.network.cifar10 import CIFAR10Net
network = CIFAR10Net(config.network)

NetworkManager.summarize(network, config.network, config.dataset)


from eva4.optimizer.optimizer_manager import OptimizerManager

optimizer = OptimizerManager.get_optimizer(network, config.optimizer)


from eva4.scheduler.scheduler_manager import SchedulerManager

scheduler = SchedulerManager.get_scheduler(optimizer, config.scheduler)


from eva4.training.training_manager import TrainingManager


training = TrainingManager(config, dataset, network, scheduler, optimizer)
training.start()

