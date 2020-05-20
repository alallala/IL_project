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
BATCH_SIZE = 128
NUM_EPOCHS = 70
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
    self.num_classes += n

    self.to(DEVICE)
    print('{} new classes'.format(len(targets)))

    #merge new data and exemplars
    for y, exemplars in enumerate(self.exemplars):
        dataset.append(exemplars, [y]*len(exemplars))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    #Store network outputs with pre-updated parameters
    q = torch.zeros(len(dataset), self.n_classes).to(DEVICE)
    for images, labels, indexes in dataloader:
        images = images.to(DEVICE)
        indexes = indexes.to(DEVICE)

        g = F.sigmoid(self(images))
        q[indexes] = g.data
    q.to(DEVICE)

    optimizer = self.optimizer

    i = 0
    for epoch in range(NUM_EPOCHS):
        for images, labels, indexes in dataloader:
            if i%5 == 0:
                print('-'*30)
                print('Epoch {}/{}'.format(i+1, NUM_EPOCHS))
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            indexes = indexes.to(DEVICE)

            #zero-ing the gradients
            optimizer.zero_grd()
            g = self(images)

            #classification Loss
            loss = self.loss(g, labels)

            #distillation Loss
            if self.num_know > 0:
                g = F.sigmoid(g)
                q_i = q[indexes]
                dist_loss = sum(self.dist_loss(g[:, y], q_i[:, y]) for y in range(self.num_known))

                loss += dist_loss

            loss.backward()
            optimizer.step()

            if i%5 == 0:
                print("Loss: {:.4f}".format(loss.data[0]))

    def reduce_examplars_set(self, m):
        for y, examplare in enumerate(self.examplars):
            self. examplar_sets[y] = P_y[:m]


    def construct_examplars_set(self):
        pass            
