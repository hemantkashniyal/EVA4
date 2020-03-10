import torch.optim as optim

from eva4.common import reset_seed, get_config, get_device

def get_optimizer(model, config):
  optimizer = None

  if config.get("optimizer") == "SGD":
    lr = config.get("learning_rate")
    momentum = config.get("momentum")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

  return optimizer