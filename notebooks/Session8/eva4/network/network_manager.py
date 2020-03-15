import torch.nn as nn
from torchsummary import summary

class NetworkManager(object):
    @classmethod
    def summarize(cls, model: nn.Module, network_config, dataset_config):
        summary(model, input_size=dataset_config.input_size)
