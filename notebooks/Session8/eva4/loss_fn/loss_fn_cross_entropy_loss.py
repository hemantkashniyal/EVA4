import torch
from torch.nn import CrossEntropyLoss


class CrossEntropyLossFn(CrossEntropyLoss):
    KEY = "CrossEntropyLoss"
    
    def __init__(self, loss_fn_config):
        self.config = loss_fn_config
        CrossEntropyLoss.__init__(self)