import torch
from torch.optim.lr_scheduler import StepLR

class StepLRScheduler(StepLR):
    KEY = "StepLR"
    
    def __init__(self, optimizer, scheduler_config):
        self.config = scheduler_config
        StepLR.__init__(self,
            optimizer,
            step_size=self.config.step_size,
            gamma=self.config.gamma,
        )