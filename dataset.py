from torchvision.datasets import CIFAR10
import numpy as np
import torch
from PIL import Image

class _CIFAR10(CIFAR10):
  
  base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
  
  def __init__(self, root, class_range=np.arange(10), train=True, transform=None, target_transform=None, download=False):
    super(_CIFAR10, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
    
    if self.train:
      train_data = []
      train_labels = []
      
      for i in range(len(self.train_data)):
        if self.train_labels[i] in class_range:
          train_data.append(self.train_data[i])
          train_labels.append(self.train_labels[i])
      
      self.train_data = np.array(train_data)
      self.train_labels = np.array(train_labels)
   
    else:
      test_data = []
      test_labels = []
      
      for i in range(len(self.test_data)):
        if self.test_labels[i] in class_range:
          test_data.append(self.test_data[i])
          test_labels.append(self.test_labels[i])
      
      self.test_data = np.array(test_data)
      self.test_labels = np.array(test_labels)
     
  
  def __getitem__(self, index):
    if self.train:
      img, target = self.train_data[index], self.train_labels[index]
    else:
      img, target = self.test_data[index], self.test_labels[index]
    
    img = Image.fromarray(img)
    
    if transform is not None:
      img = self.transform(img)
      
    if self.target_transform is not None:
            target = self.target_transform(target)

    return index, img, target
  
  def __len__(self):
    if self.train:
      return len(self.train_data)
    else:
      return len(self.test_data)
 
  def get_class_imgs(self, label):
    if self.train:
      return self.train_data[np.array(self.train_labels) == label]
    else:
      return self.test_data[np.array(self.test_labels) == label]
    
  def append(self, images, labels):
        """Append dataset with images and labels
        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """
        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels
        
class _CIFAR100(_CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    
