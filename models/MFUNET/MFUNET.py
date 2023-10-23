"""Motion Field U-Net (MF-U-net) model definition with definitions of custom loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from modelcomponents import RainNet as RN

class MFUNET(pl.LightningModule):
    """Model for the Motion Field U-Net (MF-U-net) neural network."""

    def __init__(self, config):

        super().__init__()
        self.save_hyperparameters()

        self.input_shape = config.model.rainnet.input_shape
        self.personal_device = torch.device(config.train_params.device)
        self.network = RN(
            kernel_size=config.model.rainnet.kernel_size,
            mode=config.model.rainnet.mode,
            im_shape=self.input_shape[1:],  # x,y
            conv_shape=config.model.rainnet.conv_shape,
        )

        if config.model.loss.name == "rmse":
            self.criterion = RMSELoss()
        else:
            raise NotImplementedError(f"Loss {config.model.loss.name} not implemented!")

        # on which leadtime to train the NN on?
        self.train_leadtimes = config.model.train_leadtimes
        self.verif_leadtimes = config.train_params.verif_leadtimes
        # How many leadtimes to predict
        self.predict_leadtimes = config.prediction.predict_leadtimes

        # 1.0 corresponds to harmonic loss weight decrease,
        # 0.0 to no decrease at all,
        # less than 1.0 is sub-harmonic,
        # more is super-harmonic
        discount_rate = config.model.loss.discount_rate
        # equal weighting for each lt, sum to one.
        if discount_rate == 0:
            self.train_loss_weights = (
                np.ones(self.train_leadtimes) / self.train_leadtimes
            )
            self.verif_loss_weights = (
                np.ones(self.verif_leadtimes) / self.verif_leadtimes
            )
        # Diminishing weight by n_lt^( - discount_rate), sum to one.
        else:
            train_t = np.arange(1, self.train_leadtimes + 1)
            self.train_loss_weights = (
                train_t ** (-discount_rate) / (train_t ** (-discount_rate)).sum()
            )
            verif_t = np.arange(1, self.verif_leadtimes + 1)
            self.verif_loss_weights = (
                verif_t ** (-discount_rate) / (verif_t ** (-discount_rate)).sum()
            )

        # optimization parameters
        self.lr = float(config.model.lr)
        self.lr_sch_params = config.train_params.lr_scheduler
        self.automatic_optimization = False

        # Whether to apply differencing
        self.apply_differencing = config.model.apply_differencing
        
        # Whether to apply bboxing
        self.apply_bbox = config.model.apply_bbox

    def forward(self, x):
        mf = self.network(x)
        return self._extrapolate(1, x[:,-1:], mf)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_sch_params.name is None:
            return optimizer
        elif self.lr_sch_params.name == "reduce_lr_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, **self.lr_sch_params.kwargs
            )
            return [optimizer], [lr_scheduler]
        else:
            raise NotImplementedError("Lr scheduler not defined.")

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        y_hat, total_loss = self._iterative_prediction(batch=batch, stage="train")
        opt.step()
        opt.zero_grad()
        self.log("train_loss", total_loss)
        return {"prediction": y_hat, "loss": total_loss}

    def validation_step(self, batch, batch_idx):
        y_hat, total_loss = self._iterative_prediction(batch=batch, stage="valid")
        self.log("val_loss", total_loss)
        return {"prediction": y_hat, "loss": total_loss}

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])

    def test_step(self, batch, batch_idx):
        y_hat, total_loss = self._iterative_prediction(batch=batch, stage="test")
        self.log("test_loss", total_loss)
        return {"prediction": y_hat, "loss": total_loss}
    
    def _extrapolate(self, timesteps, precip, motion_field):
        velocity = motion_field / (motion_field.shape[-1] / 2)

        x_values, y_values = torch.meshgrid(torch.arange(velocity.shape[-2]), torch.arange(velocity.shape[-1]))
        xy_coords = torch.stack([y_values, x_values]).to(self.personal_device)
        xy_coords = ((xy_coords) / ((velocity.shape[-1]) / 2) - 1)  # only works correctly for square input currently

        precip_extrap = torch.zeros((precip.shape[0], timesteps, precip.shape[2], precip.shape[3])).to(self.personal_device)
        displacement = torch.zeros((velocity.shape[0], 2, velocity.shape[2], velocity.shape[3])).to(self.personal_device)
        velocity_inc = velocity.clone()

        for ti in range(timesteps):
            coords_warped = xy_coords.unsqueeze(0) + displacement
            velocity_inc = F.grid_sample(velocity, coords_warped.movedim(1,-1), mode='bilinear', padding_mode='border', align_corners=True)
            displacement -= velocity_inc
            coords_warped = xy_coords.unsqueeze(0) + displacement
            precip_warped = F.grid_sample(precip, coords_warped.movedim(1,-1), mode='bilinear', padding_mode='zeros', align_corners=True)
            precip_extrap[:,ti:ti+1] = precip_warped

        return precip_extrap
    
    def _iterative_prediction(self, batch, stage):

        if stage == "train":
            n_leadtimes = self.train_leadtimes
            calculate_loss = True
            loss_weights = self.train_loss_weights
        elif stage == "valid" or stage == "test":
            n_leadtimes = self.verif_leadtimes
            calculate_loss = True
            loss_weights = self.verif_loss_weights
        elif stage == "predict":
            n_leadtimes = self.predict_leadtimes
            calculate_loss = False
        else:
            raise ValueError(
                f"Stage {stage} is undefined. \n choices: 'train', 'valid', test', 'predict'"
            )

        x, y, _ = batch
        x = torch.squeeze(x, 2).float()
        y = torch.squeeze(y, 2).float()
        y_seq = torch.empty(
            (x.shape[0], n_leadtimes, *self.input_shape[1:]), device=self.device
        )
        if calculate_loss:
            total_loss = 0

        for i in range(n_leadtimes):
            y_hat = self(x)
            if calculate_loss:
                y_i = y[:, None, i, :, :].clone()
                loss = self.criterion(y_hat, y_i) * loss_weights[i]
                total_loss += loss.detach()
                if stage == "train":
                    self.manual_backward(loss)
                del y_i
            y_seq[:, i, :, :] = y_hat.detach().squeeze()
            if i != n_leadtimes - 1:
                x = torch.roll(x, -1, dims=1)
                x[:, 3, :, :] = y_hat.detach().squeeze()
            del y_hat
        if calculate_loss:
            return y_seq, total_loss
        else:
            return y_seq


class RMSELoss(nn.Module):
    """RMSE loss function module.
    
    Implementation from https://discuss.pytorch.org/t/rmse-loss-function/16540/3.
    
    """

    def __init__(self, eps=1e-6):
        """Initialize loss function."""
        super().__init__()
        self.mse = nn.MSELoss()
        # Add small value to prevent nan in backwards pass
        self.eps = eps

    def forward(self, yhat, y):
        """Forward pass."""
        return torch.sqrt(self.mse(yhat, y) + self.eps)
