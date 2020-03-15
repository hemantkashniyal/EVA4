import torch
from torch.optim.lr_scheduler import MultiStepLR

class MultiStepLRScheduler(MultiStepLR):
    KEY = "MultiStepLR"
    
    def __init__(self, optimizer, scheduler_config):
        self.config = scheduler_config
        MultiStepLR.__init__(self,
            optimizer,
            milestones=self.config.milestones,
            gamma=self.config.gamma,
        )