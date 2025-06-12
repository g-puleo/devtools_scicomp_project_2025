from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import lightning.pytorch as L
import os
WHERE_TO_SAVE = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "data" ))
class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_path=WHERE_TO_SAVE, batch_size=64):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
    def prepare_data(self):
        datasets.CIFAR10(root=self.data_path, download=True)
        self.transform = transforms.Compose(
            [transforms.Resize((70, 70)), transforms.RandomCrop((64, 64)),
             transforms.ToTensor()])
    def setup(self, stage=None):
        train = datasets.CIFAR10(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=False,
        )
        self.train, self.valid = random_split(train, lengths=[45000, 5000])
        self.test = datasets.CIFAR10(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False,
        )
    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, num_workers=19)
    def val_dataloader(self):
        return DataLoader(self.valid, self.batch_size, num_workers=19)
    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, num_workers=19 )
    