import numpy as np

import torch

import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import CIFAR100

from torchvision.models import resnet18

####Hyper-parameters####
DEVICE = 'cuda' 
NUM_CLASSES = 10 
BATCH_SIZE = 128     
LR = 2       
MOMENTUM = 0.9       
WEIGHT_DECAY = 0.00001  
NUM_EPOCHS = 70      
########################


#train function
def train(net, train_dataloader):
  
  criterion = nn.CrossEntropyLoss() # for classification, we use Cross Entropy
  #criterion = nn.BCELoss()#binary CrossEntropyLoss 
  parameters_to_optimize = net.parameters() # In this case we optimize over all the parameters of AlexNet
  optimizer = optim.SGD(parameters_to_optimize, lr=LR, weight_decay=WEIGHT_DECAY)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

  net.to(DEVICE)

  i=0

  for epoch in range(70):
      
      if(i%3 == 0 ):
        print('Epoch {}/{} LR={}'.format(epoch+1, 70, scheduler.get_last_lr()))
        print('-' * 30)

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for inputs, labels in train_dataloader:
          inputs = inputs.to(DEVICE)
          labels = labels.to(DEVICE)

          net.train(True)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward      
          outputs = net(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          loss.backward()
          optimizer.step()

          # statistics
          running_loss += loss.item() * inputs.size(0)
          running_corrects += torch.sum(preds == labels.data)
      
      scheduler.step()

      epoch_loss = running_loss / len(train_dataloader.dataset)
      epoch_acc = running_corrects.double() / len(train_dataloader.dataset)

      if(i%3 == 0 ):        
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

      i+=1
  return net

#test function
def test(net, test_dataloader):
  net.to(DEVICE)
  net.train(False)

  running_corrects = 0
  for images, labels in test_dataloader:
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # Forward Pass
    outputs = net(images)

    # Get predictions
    _, preds = torch.max(outputs.data, 1)

    # Update Corrects
    running_corrects += torch.sum(preds == labels.data).data.item()

  # Calculate Accuracy
  accuracy = running_corrects / float(len(test_dataset))

  print('Test Accuracy: {}'.format(accuracy))



def main():
  
  #define images transformation
  train_transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])   

  test_transform = transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor(), 
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
  
  #creo i dataset per ora prendo solo le prime 10 classi per testare, ho esteso la classe cifar 100 con attributo 
  #classes che è una lista di labels, il dataset carica solo le foto con quelle labels
  
  range_classes = np.arange(100)
  classes_groups = np.array_split(range_classes, 10)
  print(classes_groups)

  dataset = CIFAR100(root='data/', classes=classes_groups[0], train=True, download=True, transform=train_transform)
  test_dataset = CIFAR100(root='data/', classes=classes_groups[0],  train=False, download=True, transform=test_transform)
  train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4)  
  test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

  net = resnet18(pretrained=True)
  
  #cambio il numero di classi di output
  net.fc = nn.Linear(512, 10)
  
  net = train(net, train_dataloader)
  test(net, test_dataloader)
  

if __name__ == '__main__':
  main()
