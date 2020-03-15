import torch
import random 
import numpy as np

def reset_seed(seed_value=1):
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)

  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed_value)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False