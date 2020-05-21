from iCaRL_net import iCaRL

from torchvision import transforms
from torch.utils.data import DataLoader

####Hyper-parameters####
DEVICE = 'cuda'
NUM_CLASSES = 10
BATCH_SIZE = 128
ClASSES_BATCH = 10
MEMORY_SIZE = 2000
########################

def main():

    #define images transformation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #creo i dataset per ora prendo solo le prime 10 classi per testare, ho esteso la classe cifar 100 con attributo
    #classes che è una lista di labels, il dataset carica solo le foto con quelle labels

    range_classes = np.arange(100)
    classes_groups = np.array_split(range_classes, 10)


    net = iCaRL()

    for i in (1): #range(int(100/ClASSES_BATCH)):

        train_dataset = CIFAR100(root='data/', classes=classes_groups[i], train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root='data/', classes=classes_groups[i],  train=False, download=True, transform=test_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

        iCaRL.update_representation(train_dataset)




if __name__ == '__main__':
    main()
