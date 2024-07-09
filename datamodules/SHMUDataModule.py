"""A datamodule for working with SHMU dataset for training, validation and testing that has files in h5 format."""
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import random

from datasets import SHMUDataset


class SHMUDataModule(pl.LightningDataModule):
    def __init__(self, dsconfig, train_params, predict_list="predict"):
        super().__init__()
        self.dsconfig = dsconfig
        self.train_params = train_params
        self.predict_list = predict_list

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage):
        # called on every GPU
        if stage == "fit":
            self.train_dataset = SHMUDataset(
                split="train", **self.dsconfig.SHMUDataset
            )
            self.valid_dataset = SHMUDataset(
                split="valid", **self.dsconfig.SHMUDataset
            )
        if stage == "test":
            self.test_dataset = SHMUDataset(
                split="test", **self.dsconfig.SHMUDataset
            )
        if stage == "predict":
            self.predict_dataset = SHMUDataset(
                split=self.predict_list, **self.dsconfig.SHMUDataset
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_params.train_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.train_params.valid_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_params.test_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return test_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_dataset,
            batch_size=self.train_params.predict_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return predict_loader
    
    def on_before_batch_transfer(self, batch, dataloader_idx):
        x, y, idx = batch
        if self.trainer.training:
            x, y = self.apply_augments(x, y)
        return x, y, idx
    
    def apply_augments(self, x, y):
        if self.dsconfig.SHMUDataset.augmentations.horizontal_flip:
            if random.random() >= 0.5:
                horizontal_flip = v2.RandomHorizontalFlip(1)
                x = horizontal_flip(x)
                y = horizontal_flip(y)

        if self.dsconfig.SHMUDataset.augmentations.vertical_flip:
            if random.random() >= 0.5:
                vertical_flip = v2.RandomVerticalFlip(1)
                x = vertical_flip(x)
                y = vertical_flip(y)
        
        if self.dsconfig.SHMUDataset.augmentations.rotations:
            angle = random.choice([0, 90, 180, 270])
            rotation = v2.RandomRotation((angle, angle))
            x = rotation(x)
            y = rotation(y)


        return x, y        


def _collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

