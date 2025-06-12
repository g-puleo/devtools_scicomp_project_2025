import os
from torch import optim, nn, utils, Tensor
import torch
import lightning.pytorch as L
import torchmetrics
from pyclassify.model import AlexNet
class Classifier( L.LightningModule ):
    
    def __init__( self, model ):
        super().__init__()
        self.model = model 
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=model.num_classes)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=model.num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=model.num_classes)
    def _classifier_step(self, batch):

        features = batch[0]
        true_labels = batch[1]
        logits = self(features)
        loss = nn.functional.cross_entropy(logits, true_labels)
        pred_label = torch.argmax(logits, dim=1)
        return pred_label, true_labels, loss

    def training_step(self, batch):
        pred_label, true_labels, loss = self._classifier_step(batch)
        self.log('loss', loss)
        self.train_accuracy(pred_label, true_labels)  
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False)
        return loss 
    
    def validation_step(self, batch):
        pred_label, true_labels, _ = self._classifier_step(batch)
        self.val_accuracy(pred_label, true_labels)
        self.log('validation_accuracy', self.val_accuracy, on_step=True, on_epoch=False)

    def test_step(self, batch ):
        pred_label, true_labels, _ = self._classifier_step(batch)
        self.test_accuracy(pred_label, true_labels)
        self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=False)

    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer