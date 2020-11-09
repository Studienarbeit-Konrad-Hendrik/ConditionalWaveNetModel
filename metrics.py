import torch
import torch.nn as nn

def accuracy(predicted, target):
  _, pred_pos = torch.max(predicted, dim=1)
  
  