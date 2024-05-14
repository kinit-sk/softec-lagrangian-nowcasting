"""Euclidean persistence model definition."""

import torch
import pytorch_lightning as pl

class EuclidPersistence(pl.LightningModule):
    """Model for the RainNet iterative neural network."""

    def __init__(self, config):

        super().__init__()
        self.save_hyperparameters()

        self.predict_leadtimes = config.prediction.predict_leadtimes

    def forward(self, x):
        return x[:,-1:].expand(-1, 6, -1, -1, -1)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Get data
        x, y, idx = batch

        x = torch.squeeze(x, 2)

        y_seq = x[:,-1:].expand(-1, self.predict_leadtimes, -1, -1)

        # Transform from scaled to mm/hh
        invScaler = self.trainer.datamodule.predict_dataset.invScaler
        y_seq = invScaler(y_seq)

        y_seq[y_seq < 0] = 0
        
        # Transform from mm/h to dBZ
        y_seq = self.trainer.datamodule.predict_dataset.from_transformed(
            y_seq, scaled=False
        )

        return y_seq
