import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image

from resnet import resnet18

####Hyper-parameters####
LR = 2
WEIGHT_DECAY = 0.00001       
########################


class iCarL(nn.Module):
  __init__(self, num_classes):
    super(iCarl,self).__init__()
    self.feature_extractor = resnet18()
    
    self.loss = nn.CrossEntropyLoss()
    self.dist_loss = nn.BCELoss()
    
    self.optimizer = optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    self.num_classes = num_classes
    self.num_know = 0
 
  def forward(self, x):
    x = self.feature_extractor(x)
    return(x)
  
  
    
