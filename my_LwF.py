import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange
import utils
from resnet import resnet32

DEVICE = 'cuda'
NUM_CLASSES = 10
BATCH_SIZE = 128
CLASSES_BATCH = 10
MEMORY_SIZE = 2000
WEIGHT_DECAY = 0.00005
LR = 2
NUM_EPOCHS = 70

class LwF(nn.Module):
    
    def __init__(self, feature_size):
        super(LwF,self).__init__()
      
        self._device = 0
        self._opt_name = 'sgd'
        self._lr =LR
        self._weight_decay = WEIGHT_DECAY
        self._n_epochs = NUM_EPOCHS

        self._scheduling = [49,63]
        
        self._lr_decay = 5
        
        self.features_extractor = resnet32(num_classes=feature_size)
        
        self._temperature = 1

        self._n_classes = 0

        self.known = 0

        self._add_n_classes(10, convnet=resnet32())

        self.task_size = CLASSES_BATCH

        self.to(self._device)

    def forward(self, inputs):
        x = self._features_extractor(inputs)
        x = self._classifier(x)
        return x


    def _before_task(self, data_loader):
        """Computes predictions with previous model's weights & set up optimizer."""

        if self.n_classes == 0:
            self._previous_preds = None
        else:
            print("Computing previous predictions...")
            self._previous_preds = self._compute_predictions(data_loader)
            self._add_n_classes(self._task_size)

        self._optimizer = optim.Adam(self.parameters(),self._lr,self._weight_decay)

        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                               self._scheduling,
                                                               gamma=self._lr_decay)

    def _train_task(self, train_loader, val_loader=None):

        val_acc = 0.

        prog_bar = trange(self._n_epochs, desc="Losses.")

        for epoch in prog_bar:
            avg_clf_loss, avg_distil_loss = 0., 0.
            c = 0

            self._scheduler.step()

            for i, (inputs, target, idxes) in enumerate(train_loader, start=1):

                self._optimizer.zero_grad()

                c += len(idxes)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                idxes = idxes.to(self._device)
                logits = self.forward(inputs)

                clf_loss, distil_loss = self._compute_loss(
                    logits,
                    targets,
                    idxes,
                )

                if not utils._check_loss(clf_loss) or not utils._check_loss(distil_loss):
                    import pdb
                    pdb.set_trace()

                loss = clf_loss + distil_loss

                loss.backward()
                self._optimizer.step()

                avg_clf_loss += clf_loss.item()
                avg_distil_loss += distil_loss.item()

                if i % 10 == 0:
                    prog_bar.set_description("Clf loss: {}; Distill loss: {}; Val acc: {}".format(
                        round(clf_loss.item(), 3), round(distil_loss.item(), 3), round(val_acc, 3)))

                if val_loader is not None and i % 100 == 0:
                    val_acc = self._accuracy(self._classifier(val_loader))

            prog_bar.set_description("Clf loss: {}; Distill loss: {}; Val acc: {}".format(
                round(clf_loss.item(), 3), round(distil_loss.item(), 3), round(val_acc, 3)))

    # to invoke after each iteration on the test set

    def _eval_task(self, data_loader):
        return self._classify(data_loader)


    def _add_n_classes(self, n, convnet=None):

        # first iteration
        if self._n_classes == 0:
            
            self._classifier = nn.Linear(self.features_extractor.out_dim, n, bias=False)
            torch.nn.init.kaiming_normal_(self._classifier.weight)

        else:
        # other iterations
        # save weights of old nodes
            weights = self._classifier.weight.data
        #add 10 new out features
            self._classifier = nn.Linear(self._features_extractor.out_dim,
                                         self._n_classes + n,
                                         bias=False).to(self._device)
        #initialize the new nodes weights and re-load the old weights for old nodes
            torch.nn.init.kaiming_normal_(self._classifier.weight)
            self._classifier.weight.data[:self._n_classes] = weights

        self._n_classes += n


    def _compute_predictions(self, data_loader):

        logits = torch.zeros(len(data_laoder.dataset), self._n_classes, device=self._device)

        for idxes, inputs, _ in data_loader:
            inputs = inputs.to(self._device)
            idxes = idxes[1].to(self._device)

            logits[idxes] = self.forward(inputs).detach() ** (1 / self._temperature)

        return logits

    def _compute_loss(self, logits, targets, idxes):

        # first iteration 
        if self.known == 0:
            clf_loss = F.cross_entropy(logits, targets)
            distil_loss = torch.zeros(1, device=self._device)
            
        else:
            for i in range(self._new_task_index, self._n_classes):
                targets[targets == i] = -1

            clf_loss = F.cross_entropy(logits[..., self._new_task_index:], targets, ignore_index=-1)

            distil_loss = F.binary_cross_entropy(
                F.softmax(logits[..., :self._new_task_index], dim=1), self._previous_preds[idxes])

        self.known += self.task_size
        return clf_loss, distil_loss

    
    def _classify(self, data_loader):
        ypred = []
        ytrue = []

        for inputs, targets, _ in data_loader:
            inputs = inputs.to(self._device)

            probs = self.forward(inputs).detach()
            preds = F.softmax(probs, dim=1).argmax(dim=1).cpu().numpy().tolist()

            ypred.extend(preds)
            ytrue.extend(targets)

        return (np.array(ypred) == np.array(ytrue)).sum() / len(ytrue)

    @staticmethod
    def _accuracy(self, ypred, ytrue):
        return (ypred == ytrue).sum() / len(ytrue)

    
