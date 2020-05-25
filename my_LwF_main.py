from my_LwF import LwF


from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import CIFAR100

import numpy as np
from numpy import random



####Hyper-parameters####
DEVICE = 'cuda'
BATCH_SIZE = 128
CLASSES_BATCH = 10
MEMORY_SIZE = 2000
########################


def main():
    #  Define images transformation
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),                    
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                      #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                     ])

    test_transform = transforms.Compose([transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                     #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                    ])


    print("\n")


    total_classes = 100    

    perm_id = np.random.permutation(total_classes)
    all_classes = np.arange(total_classes)
    
    '''
    #mix the classes indexes
    for i in range(len(all_classes)):
      all_classes[i] = perm_id[all_classes[i]]

    #Create groups of 10
    #classes_groups = np.array_split(all_classes, 10)
    #print(classes_groups)

    #num_iters = total_classes//CLASSES_BATCH      
    # Create class map

    class_map = {}
    #takes 10 new classes randomly
    for i, cl in enumerate(all_classes):
        class_map[cl] = i
    print (f"Class map:{class_map}\n")     
    
    # Create class map reversed
    map_reverse = {}
    for cl, map_cl in class_map.items():
        map_reverse[map_cl] = int(cl)
    print (f"Map Reverse:{map_reverse}\n")
    '''

    # Create Network
    net = LwF(self)
 
      
    #iterating until the net knows total_classes with 10 by 10 steps 

    for s in range(0, total_classes, CLASSES_BATCH):  #c'era (0, num_iter,CLASSES_BATCH), modificato perchè altrimenti avevamo num_iter=10
                                                      #CLASSES_BATCH= 10 quindi s andava da 0 a 10 e si fermava
                                                      #ora s parte da zero, salta di 10 in 10, fino ad arrivare a 100.. in totale fa 10 iter
   
       
        print(f"ITERATION: {(s//CLASSES_BATCH)+1} / {total_classes//CLASSES_BATCH}\n")
      
        print("\n")
                
        # Load Datasets    
        
                                                             #train data_loader loads images in classes [0:10] then in [10:20] etc..          
        train_dataset = CIFAR100(root='data',train=True,classes=all_classes[s:s+CLASSES_BATCH],download=True,transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=4)
                                                             #test data_loader loades images in classes [0:10] then [0:20] etc..
        test_dataset = CIFAR100(root='data',train=False,classes=all_classes[:s+CLASSES_BATCH],download=True, transform=test_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False, num_workers=4)
        
        

        net._before_task(train_dataloader)

        net._added_n_classes(CLASSES_BATCH)
 
        net.train()
      
        net._train_task(train_dataloader)

        net.eval()
        
        net.eval(test_dataloader)


if __name__ == '__main__':
    main()
