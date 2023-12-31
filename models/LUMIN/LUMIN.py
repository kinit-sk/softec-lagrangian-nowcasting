"""Lagrangian U-Net Model for Iterative Nowcasting (LUMIN) model definition with definitions of custom loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

from modelcomponents import RainNet as RN

class LUMIN(pl.LightningModule):
    """Model for the Lagrangian U-Net Model for Iterative Nowcasting (LUMIN) neural network."""

    def __init__(self, config):

        super().__init__()
        self.save_hyperparameters()

        self.input_shape = config.model.mfunet.input_shape
        self.personal_device = torch.device(config.train_params.device)
        self.mfunet_network = RN(
            kernel_size=config.model.mfunet.kernel_size,
            mode=config.model.mfunet.mode,
            im_shape=config.model.mfunet.input_shape[1:],  # x,y
            conv_shape=config.model.mfunet.conv_shape,
        )

        self.advf_network = RN(
            kernel_size=config.model.advfunet.kernel_size,
            mode=config.model.advfunet.mode,
            im_shape=config.model.advfunet.input_shape[1:],  # x,y
            conv_shape=config.model.advfunet.conv_shape,
        )

        if config.model.loss.name == "rmse":
            self.criterion = RMSELoss()
        elif config.model.loss.name == "mse":
            self.criterion = nn.MSELoss()
        elif config.model.loss.name == "logcosh":
            self.criterion = LogCoshLoss()
        else:
            raise NotImplementedError(f"Loss {config.model.loss.name} not implemented!")
        
        if config.model.loss.regularized == True:
            self.criterion = ConservationLawRegularizationLoss(self.criterion, **config.model.loss.kwargs)
        

        # on which leadtime to train the NN on?
        self.train_leadtimes = config.model.train_leadtimes
        self.verif_leadtimes = config.train_params.verif_leadtimes
        # How many leadtimes to predict
        self.predict_leadtimes = config.prediction.predict_leadtimes

        self.apply_differencing = config.model.apply_differencing

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

    def forward(self, x, x_diff):
        # Fist stage - Motion-Field U-Net
        mf = self.mfunet_network(x)
        extrapolated = self._extrapolate(1, x_diff[:,-1:], mf)

        # Second stage - Advection-free U-Net
        x_lagrangian = x_diff.clone()

        for i in range(x_diff.shape[1]):
            x_lagrangian[:,i] = self._extrapolate(x_lagrangian.shape[1]-i, x_lagrangian[:,[i]], mf)[:,-1]

        result = self.advf_network(x_lagrangian)

        return result, extrapolated, mf

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
        y_hat, loss = self._iterative_prediction(batch=batch, stage="train")
        opt.step()
        opt.zero_grad()
        self.log("train_loss", loss["total_loss"])
        if isinstance(self.criterion, ConservationLawRegularizationLoss):
            self.log("train_crit_loss", loss["total_crit_loss"])
            self.log("train_phys_loss", loss["total_phys_loss"])
            self.log("train_extra_crit_loss", loss["total_extra_crit_loss"])
        return {"prediction": y_hat, "loss": loss["total_loss"]}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            y_hat, loss = self._iterative_prediction(batch=batch, stage="valid")
        self.log("val_loss", loss["total_loss"])
        return {"prediction": y_hat, "loss": loss["total_loss"]}

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            y_hat, loss = self._iterative_prediction(batch=batch, stage="test")
        self.log("test_loss", loss["total_loss"])
        return {"prediction": y_hat, "loss": loss["total_loss"]}
    
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

        x, x_diff, y, _ = batch
        x = torch.squeeze(x, 2).float()
        x_diff = torch.squeeze(x_diff, 2).float()
        y = torch.squeeze(y, 2).float()
        y_seq = torch.empty(
            (x.shape[0], n_leadtimes, *self.input_shape[1:]), device=self.device
        )
        if calculate_loss:
            total_loss = 0
            if isinstance(self.criterion, ConservationLawRegularizationLoss):
                total_crit_loss = 0
                total_extra_crit_loss = 0
                total_phys_loss = 0

        for i in range(n_leadtimes):
            y_hat, y_extra, mf = self(x, x_diff)
            if calculate_loss:
                y_i = y[:, None, i, :, :].clone()
                if isinstance(self.criterion, RMSELoss):
                    loss = self.criterion(y_hat, y_i) * loss_weights[i] + self.criterion(y_hat, y_extra) * loss_weights[i]
                elif isinstance(self.criterion, ConservationLawRegularizationLoss):
                    loss, crit_loss, extra_crit_loss, phys_loss = self.criterion(y_hat, y_extra, y_i, mf, stage=stage)
                    loss, crit_loss, extra_crit_loss, phys_loss = map(lambda l: torch.mul(l, loss_weights[i]), (loss, crit_loss, extra_crit_loss, phys_loss))
                    total_crit_loss += crit_loss.detach()
                    total_extra_crit_loss += extra_crit_loss.detach()
                    total_phys_loss += phys_loss.detach()
                total_loss += loss.detach()
                if stage == "train":
                    self.manual_backward(loss)
                del y_i
            y_seq[:, i, :, :] = y_hat.detach().squeeze()
            if i != n_leadtimes - 1:
                x = torch.roll(x, -1, dims=1)
                x[:, -1, :, :] = y_hat.detach().squeeze()
            del y_hat
        if calculate_loss and isinstance(self.criterion, RMSELoss):
            return y_seq, {"total_loss": total_loss}
        elif calculate_loss and isinstance(self.criterion, ConservationLawRegularizationLoss):
            return y_seq, {"total_loss": total_loss, "total_crit_loss": total_crit_loss, "total_extra_crit_loss": total_extra_crit_loss, "total_phys_loss": total_phys_loss}
        else:
            return y_seq
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Get data
        x, x_diff, y, _, first_x = batch

        # Perform prediction with LCNN model
        x_ = x.clone()
        y_seq = self._iterative_prediction(batch=(x, x_diff, y, first_x), stage="predict")

        # Transform from scaled to mm/hh
        invScaler = self.trainer.datamodule.predict_dataset.invScaler
        y_seq = invScaler(y_seq)
        x = invScaler(x_)

        # If we applied differencing, integrate back to full fields
        if self.apply_differencing:
            # This should be a dummy transform that has no effect (since scaling should be "none"),
            # but keep to match what is done to predictions
            first_x = invScaler(first_x)
            # Integrate observation back to full fields
            temp = torch.permute(first_x, dims=(0, 3, 1, 2)) + x[:, 0, :, :]
            x[:, 0, :, :] = temp
            for i in range(1, x.shape[1]):
                x[:, i, :, :] = x[:, i, :, :] + x[:, i - 1, :, :]
            # First prediction with last observation
            temp = y_seq[:, 0, :, :] + x[:, -1, 0, :, :]
            y_seq[:, 0, :, :] = temp
            # Iterate through rest of predictions
            for i in range(1, self.predict_leadtimes):
                y_seq[:, i, :, :] = y_seq[:, i - 1, :, :] + y_seq[:, i, :, :]

            # Set negative values (in mm/h) to 0
            y_seq[y_seq < 0] = 0
        
        # Transform from mm/h to dBZ
        y_seq = self.trainer.datamodule.predict_dataset.from_transformed(
            y_seq, scaled=False
        )

        del x
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

import torchvision.transforms.functional as TF

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + nn.functional.softplus(-2. * x) - math.log(2.0)
    
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)

class ConservationLawRegularizationLoss(nn.Module):
    def __init__(self, base_criterion, beta=0.5, gamma=0.5):
        super(ConservationLawRegularizationLoss, self).__init__()
        self.sobel_x = torch.tensor([[-1.,  0.,  1.],
                                     [-2.,  0.,  2.],
                                     [-1.,  0.,  1.]]).view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        self.sobel_y = torch.tensor([[-1., -2., -1.],
                                     [ 0.,  0.,  0.],
                                     [ 1.,  2.,  1.]]).view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        self.beta = beta
        self.gamma = gamma
        self.criterion = base_criterion

    def forward(self, target, extrapolated, final_output, motion_field, stage):
        criterion_loss = self.criterion(TF.center_crop(final_output, 336-48), TF.center_crop(target, 336-48))
        extra_criterion_loss = self.criterion(TF.center_crop(extrapolated, 336-48), TF.center_crop(target, 336-48))

        device = final_output.device
        
        # physics-informed conservation of mass loss
        diff_u = F.conv2d(motion_field[:,0:1], self.sobel_x.to(device))
        diff_v = F.conv2d(motion_field[:,1:2], self.sobel_y.to(device))
        physics_loss = torch.sum(torch.abs(diff_u + diff_v)) / (motion_field.shape[0] * motion_field.shape[2] * motion_field.shape[3])

        if stage == "valid" or stage == "test":
            return criterion_loss, criterion_loss, extra_criterion_loss, physics_loss
        
        criterion_loss_weighted = (1 - self.beta) * (1 - self.gamma) * criterion_loss
        extra_criterion_loss_weighted = (1 - self.beta) * (self.gamma) * extra_criterion_loss
        physics_loss_weighted = (self.beta) * physics_loss

        return criterion_loss_weighted + extra_criterion_loss_weighted + physics_loss_weighted, criterion_loss, extra_criterion_loss, physics_loss