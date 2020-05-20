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
ClASSES_BATCH =10
LR = 2
REDUCE_EPOCHS = [
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
  
  for i in range(100/BATCH_CLASSES):
    #cambio il numero di classi di output
    net.fc = nn.Linear(512, 10+i*10)
    
    if i != 0:
      
      #creating dataset for current iteration
      train_dataset = CIFAR100(root='data/', classes=classes_groups[i], train=True, download=True, transform=train_transform)
      test_dataset = CIFAR100(root='data/', classes=classes_groups[i],  train=False, download=True, transform=test_transform)
      
      #creating dataset for test on previous classes
      previous_classes = np.array([])
      for j in range(i-1):
        np.concatenate((previous_classes), classes_groups[j])
      test_prev_dataset = CIFAR100(root='data/', classes=previous_classes,  train=False, download=True, transform=test_transform)
      
      #creating dataset for all classes
      all_classes = np.concatenate((previous_classes, classes_groups[i]))
      test_all_dataset = CIFAR100(root='data/', classes=all_classes,  train=False, download=True, transform=test_transform)
    
      #creating dataloaders      
      train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)  
      test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
      test_prev_dataloader = DataLoader(test_prev_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
      test_all_dataloader = DataLoader(test_all_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

       
      net = train(net, train_dataloader)
      print('Test on new classes')
      test(net, test_dataloader)
      print('Test on old classes')
      test(net, test_prev_dataloader)
      print('Test on all classes')
      test(net, test_all_dataloader)
  
    else:
      train_dataset = CIFAR100(root='data/', classes=classes_groups[i], train=True, download=True, transform=train_transform)
      test_dataset = CIFAR100(root='data/', classes=classes_groups[i],  train=False, download=True, transform=test_transform)
      train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)  
      test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
      net = train(net, train_dataloader)
      print('Test on first 10 classes')
      test(net, test_dataloader)

if __name__ == '__main__':
    main()
