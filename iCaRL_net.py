import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from resnet import resnet18

####Hyper-parameters####
LR = 2
WEIGHT_DECAY = 0.00001
BATCH_SIZE =128
DEVICE = 'cuda'
########################


class iCarL(nn.Module):
  __init__(self, num_classes):
    super(iCarl,self).__init__()
    self.feature_extractor = resnet18()
    self.feature_extractor.fc = nn.Linear(64, num_classes)

    self.loss = nn.CrossEntropyLoss()
    self.dist_loss = nn.BCELoss()

    self.optimizer = optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    self.num_classes = num_classes
    self.num_know = 0
    self.examplars = []



  def forward(self, x):
    x = self.feature_extractor(x)
    return(x)

  def update_representation(self, dataset):
    targets = list(set(self.targets))

    #Increment classes
    in_features = self.feature_extractor.fc.in_features
    out_features = self.feature_extractor.fc.out_features
    weight = self.feature_extractor.fc.weight.data
    bias = self.feature_extractor.fc.bias.data

    self.feature_extractor.fc = nn.Linear(in_features, out_features+n, bias=False)
    self.feature_extractor.fc.weight.data[:out_features] = weight
    self.feature_extractor.fc.bias.data[:out_features] = bias
    self.n_classes += n

    self.to(DEVICE)
    print('{} new classes'.format(len(targets)))

    #merge new data and exemplars
    for y, exemplars in enumerate(self.exemplars):
        dataset.append(exemplars, [y]*len(exemplars))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    #Store network outputs with pre-updated parameters
    q = torch.zeros(len(dataset), self.n_classes).to(DEVICE)
    for indice
