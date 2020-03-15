import torch
from torch.nn import NLLLoss


class NLLLossFn(NLLLoss):
    KEY = "NLLLoss"
    
    def __init__(self, loss_fn_config):
        self.config = loss_fn_config
        NLLLoss.__init__(self)