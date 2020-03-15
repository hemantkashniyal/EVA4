import torch
import json
from environs import Env

class TrainingConfig(object):
    def __init__(self, env=None):
        if env is None:
            env = Env()
            env.read_env()

        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_available else "cpu"
        self.epochs = env.int("APP_TRAINING_EPOCHS")
        self.desired_accuracy = env.float("APP_TRAINING_DESIRED_ACCURACY")
        self.break_on_reaching_desired_accuracy = env.bool("APP_TRAINING_BREAK_ON_REACHING_DESIRED_ACCURACY")
        self.consecutive_desired_accuracy = env.int("APP_TRAINING_CONSECUTIVE_DESIRED_ACCURACY")

        self.dataset = self._get_dataset_config(env)
        self.scheduler = self._get_scheduler_config(env)
        self.optimizer = self._get_optimizer_config(env)
        self.regularizer = RegularizerConfig(env)
   
        self.loss_fn = LossFnConfig(env)
        self.network = NetworkConfig(self.dataset, env)
        
    def dict(self):
        json_str = json.dumps(self, default=lambda x: x.__dict__, sort_keys=False, indent=4)
        return json.loads(json_str)

    @classmethod
    def print(cls, config):
        print("Config: ", json.dumps(config, default=lambda x: x.__dict__, sort_keys=False, indent=4))

    def _get_dataset_config(self, env):
        datasets = {
            "CIFAR10": CIFAR10DatasetConfig,
        }
        
        return datasets.get(env("APP_DATASET"))(env)

    def _get_scheduler_config(self, env):
        schedulers = {
            "None": None,
            "StepLR": StepLRSchedulerConfig,
            "MultiStepLR": MultiStepLRSchedulerConfig,
        }
        
        scheduler = schedulers.get(env("APP_SCHEDULER"))
        if scheduler:
            return scheduler(env)        
        return None

    def _get_optimizer_config(self, env):
        optimizers = {
            "None": None,
            "SGD": SGDOptimizerConfig,
        }
        
        optimizer = optimizers.get(env("APP_OPTIMIZER"))
        if optimizer:
            return optimizer(env)        
        return None

class LossFnConfig(object):
    def __init__(self, env=None):
        if env is None:
            env = Env()
            env.read_env()

        self.type =  env("APP_LOSS_FN")

    def dict(self):
        json_str = json.dumps(self, default=lambda x: x.__dict__, sort_keys=False, indent=4)
        return json.loads(json_str)

class NetworkConfig(object):
    def __init__(self, dataset_config, env=None):
        if env is None:
            env = Env()
            env.read_env()

        self.dropout = env.float("APP_NETWORK_DROPOUT")
        self.bias_enabled = env.bool("APP_NETWORK_BIAS_ENABLED")
        self.input_channel = dataset_config.input_channel

    def dict(self):
        json_str = json.dumps(self, default=lambda x: x.__dict__, sort_keys=False, indent=4)
        return json.loads(json_str)

class RegularizerConfig(object):
    def __init__(self, env=None):
        if env is None:
            env = Env()
            env.read_env()

        self.l1_enabled = env.bool("APP_REGULARIZER_L1_ENABLED", False)
        if self.l1_enabled:
            self.l1_lambda = env.float("APP_REGULARIZER_L1_LAMBDA")

        self.l2_enabled = env.bool("APP_REGULARIZER_L2_ENABLED", False)
        if self.l2_enabled:
            self.l2_lambda = env.float("APP_REGULARIZER_L2_LAMBDA")

    def dict(self):
        json_str = json.dumps(self, default=lambda x: x.__dict__, sort_keys=False, indent=4)
        return json.loads(json_str)
        
class SGDOptimizerConfig(object):
    def __init__(self, env=None):
        if env is None:
            env = Env()
            env.read_env()

        self.type = "SGD"
        self.learning_rate = env.float("APP_OPTIMIZER_SGD_LEARNING_RATE")
        self.momentum = env.float("APP_OPTIMIZER_SGD_MOMENTUM")
        self.weight_decay = env.float("APP_OPTIMIZER_SGD_WEIGHT_DECAY", 0)
        if env.bool("APP_REGULARIZER_L2_ENABLED"):
            self.weight_decay = env.float("APP_REGULARIZER_L2_LAMBDA")

    def dict(self):
        json_str = json.dumps(self, default=lambda x: x.__dict__, sort_keys=False, indent=4)
        return json.loads(json_str)

class StepLRSchedulerConfig(object):
    def __init__(self, env=None):
        if env is None:
            env = Env()
            env.read_env()

        self.type = "StepLR"
        self.step_size = env.int("APP_SCHEDULER_STEPLR_STEP_SIZE")
        self.gamma = env.float("APP_SCHEDULER_STEPLR_GAMMA")

    def dict(self):
        json_str = json.dumps(self, default=lambda x: x.__dict__, sort_keys=False, indent=4)
        return json.loads(json_str)


class MultiStepLRSchedulerConfig(object):
    def __init__(self, env=None):
        if env is None:
            env = Env()
            env.read_env()

        self.type = "MultiStepLR"
        self.milestones = env.list("APP_SCHEDULER_MULTISTEPLR_MILESTONES", subcast=int)
        self.gamma = env.float("APP_SCHEDULER_MULTISTEPLR_GAMMA")

    def dict(self):
        json_str = json.dumps(self, default=lambda x: x.__dict__, sort_keys=False, indent=4)
        return json.loads(json_str)

class CIFAR10DatasetConfig(object):
    def __init__(self, env=None):
        if env is None:
            env = Env()
            env.read_env()

        self.type = "CIFAR10"
        input_size = (3, 32, 32)
        input_channel = input_size[0]
        input_dimension = input_size[1:]

        batch_size = env.int("APP_DATASET_DATA_LOADER_BATCH_SIZE")
        num_workers = 4
        pin_memory = True
        if not torch.cuda.is_available():
            batch_size = 64
            num_workers = 1
            pin_memory = True

        self.dataset_root = env.str("APP_DATASET_ROOT")
        self.batch_size = batch_size
        self.shuffle = env.bool("APP_DATASET_DATA_LOADER_SHUFFLE")

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.input_size = input_size
        self.input_channel = input_channel
        self.input_dimension = input_dimension
        
        self.dataset_std = (0.247, 0.2435, 0.2616)
        self.dataset_mean = (0.4914, 0.4822, 0.4465)

    def dict(self):
        json_str = json.dumps(self, default=lambda x: x.__dict__, sort_keys=False, indent=4)
        return json.loads(json_str)


