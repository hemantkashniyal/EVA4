import torch
from torch.optim import SGD


class SGDOptimizer(SGD):
    KEY = "SGD"
    
    def __init__(self, model, optimizer_config):
        self.config = optimizer_config
        SGD.__init__(self,
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
        )