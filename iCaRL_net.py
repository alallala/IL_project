import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from resnet import resnet32

####Hyper-parameters####
LR = 2
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 128
NUM_EPOCHS = 70
DEVICE = 'cuda'
########################


class iCaRL(nn.Module):
  def __init__(self, num_classes):
    super(iCaRL,self).__init__()
    self.feature_extractor = resnet32()
    self.feature_extractor.fc = nn.Linear(64, num_classes)

    self.loss = nn.CrossEntropyLoss()
    self.dist_loss = nn.BCEWithLogitsLoss()

    self.optimizer = optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    self.num_classes = num_classes
    self.num_known = 0
    self.exemplars = []



  def forward(self, x):
    x = self.feature_extractor(x)
    return(x)

  def update_representation(self, dataset):

    targets = list(set(dataset.targets))
    n = len(targets)
    self.to(DEVICE)
    print('{} new classes'.format(len(targets)))

    #merge new data and exemplars
    for y, exemplars in enumerate(self.exemplars):
        dataset.append(exemplars, [y]*len(exemplars))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    #Store network outputs with pre-updated parameters
    q = torch.zeros(len(dataset), self.num_classes).to(DEVICE)
    for images, labels, indexes in dataloader:
        images = images.to(DEVICE)
        indexes = indexes.to(DEVICE)
        q[indexes] = self(images)
    q.to(DEVICE)


    #Increment classes
    in_features = self.feature_extractor.fc.in_features
    out_features = self.feature_extractor.fc.out_features
    weight = self.feature_extractor.fc.weight.data

    self.feature_extractor.fc = nn.Linear(in_features, out_features+n, bias=False)
    self.feature_extractor.fc.weight.data[:out_features] = weight
    self.num_known = self.num_classes
    self.num_classes += n

    optimizer = self.optimizer

    i = 0
    self.to(DEVICE)
    for epoch in range(NUM_EPOCHS):
        if i%5 == 0:
            print('-'*30)
            print('Epoch {}/{}'.format(i+1, NUM_EPOCHS))
        for images, labels, indexes in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            indexes = indexes.to(DEVICE)


            #zero-ing the gradients
            optimizer.zero_grad()
            out = self(images)
            #classification Loss
            loss = self.loss(out, labels)
            #distillation Loss
            if self.num_known > 0:
                q_i = q[indexes]
                dist_loss = sum(self.dist_loss(out[:, y], q_i[:, y]) for y in range(self.num_known))

                loss += dist_loss

            loss.backward()
            optimizer.step()

        if i%5 == 0:
            print("Loss: {:.4f}".format(loss.item()))
        i+=1

  def reduce_exemplars_set(self, m):
    for y, exemplars in enumerate(self.exemplars):
        self.exemplars[y] = exemplars[:m]


  def construct_exemplars_set(self, images, m):

    feature_extractor = self.feature_extractor.to(DEVICE)
    features = []
    for img in images:
        img = img.unsqueeze(0)
        img = img.to(DEVICE)
        feature = feature_extractor.extract_features(img).data.cpu().numpy()
        features.append(feature)

    class_mean = np.mean(np.array(features))

    print('LUNGHEZA IMMAGINI')
    print(len(images))
    print('LUNGHEZZA FEATURES')
    print(len(features))

    exemplar_set = []
    exemplar_features = []
    for k in range(m):
        exemplar_sum = np.sum(exemplar_features)
        candidates = (features + exemplar_sum)*1.0/(k+1)
        candidates = np.sqrt([(class_mean-c)**2 for c in candidates])

        print('LUNGHEZZA POSSIBILI')
        print(len(candidates))
        i = np.argmin(candidates)
        print('Indice scelto:{}'.format(i))

        exemplar_set.append(images[i])
        exemplar_features.append(features[i])
        """
        print('here')
        features = np.delete(features, i)
        images = np.delete(images.numpy())
        print('after')
        """
    self.exemplars.append(exemplar_set)

  #da cambiare completamente
  '''def classify(self, x):
    #computing exemplars mean
    _means=[]
    for exemplars in self.exemplars:
        features = []
        for ex in exemplars:
            features.append(self.feature_extractor.extract_features(ex))
        exemplars_mean.append(np.mean(features))

    feature = self.feature_extractor(x)

    distances = np.sqrt([(feature - mean)**2 for mean in exemplars_mean])
    pred = np.argmin(distances)'''
    
    def classify(self, dataloader):
       
        #compute the mean for each examplars 
        exemplar_means=[]
        for exemplars in self.exemplars:
            features = []
        
        for ex in exemplars:
            features.append(self.feature_extractor.extract_features(ex))
        exemplar_means.append(np.mean(features))
        
        if exemplar_means is None:
            raise ValueError(
                "Cannot classify without built examplar means,"
            )
        
        if exemplar_means.shape[0] != self._n_classes:
            raise ValueError(
                "The number of examplar means ({}) is inconsistent".format(exemplar_means.shape[0])"
            )

        ypred = []
        ytrue = []

        for _, inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)
            #compute the feature map of the input 
            features = self.feature_extractor(inputs)
            
            pred_labels = []
            
            for feature in features:
              #computing L2 distance
              distances = torch.pow(examplar_mean - feature, 2).sum(-1)
              pred_labels.append(distances.argmin().item())
              
            preds = np.array(pred_labels)

            ypred.extend(preds)
            ytrue.extend(targets)

        return np.array(ypred), np.array(ytrue)

  
