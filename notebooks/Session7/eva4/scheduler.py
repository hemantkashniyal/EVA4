from torch.optim.lr_scheduler import StepLR, MultiStepLR

from eva4.common import reset_seed, get_config, get_device

def get_scheduler(optimizer, config):
  scheduler = None

  if config.get("scheduler") == "StepLR":
    step_size = config.get("step_size")
    gamma = config.get("gamma")
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

  if config.get("scheduler") == "MultiStepLR":
    milestones = config.get("milestones")
    gamma = config.get("gamma")
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

  return scheduler