import torch

class L1Regularizer(object):
    KEY = "L1"

    def __init__(self, regularizer_config):
        self.config = regularizer_config
        self.l1_lambda = self.config.l1_lambda

    def regularize(self, model, loss):
        reg_loss = 0.
        for param in model.parameters():
            reg_loss += torch.sum(param.abs())
        return loss + (self.l1_lambda * reg_loss)