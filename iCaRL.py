import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image

from resnet import resnet18


class iCarL(nn.Module):
  __init__(self, num_classes):
    super(iCarl,self).__init__()
    self.feature_extractor = resnet18()
    
