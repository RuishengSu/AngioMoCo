import os
from pathlib import Path
from typing import Any, List
import numpy as np
import torch
import torchfields
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric, MeanSquaredError
from torchmetrics.functional import structural_similarity_index_measure
import time
from src.models.components import losses, ndutils
from src.models.components.utils_mrclean import visualize_series, visualize_pair


class MNISTLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            loss: losses.VMLoss,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "loss"])

        self.net = net
        # loss function
        self.loss = loss

        # metric objects for calculating and averaging accuracy across batches
        self.train_mse_dsa = MeanSquaredError()
        self.val_mse_dsa = MeanSquaredError()
        self.test_mse_dsa = MeanSquaredError()

        self.train_mse_dla_warp_mask = MeanSquaredError()
        self.val_mse_dla_warp_mask = MeanSquaredError()
        self.test_mse_dla_warp_mask = MeanSquaredError()
        self.train_mse_dla_warp_contrast = MeanSquaredError()
        self.val_mse_dla_warp_contrast = MeanSquaredError()
        self.test_mse_dla_warp_contrast = MeanSquaredError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_ssim_loss = MeanMetric()
        self.val_ssim_loss = MeanMetric()
        self.test_ssim_loss = MeanMetric()

        self.train_smooth_loss = MeanMetric()
        self.val_smooth_loss = MeanMetric()
        self.test_smooth_loss = MeanMetric()

        self.series_proc_times = []
        self.frame_proc_times = []

        # for tracking best so far validation metric
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor, y: torch.Tensor, registration: bool = False):
        return self.net(x, y, registration)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_loss_best doesn't store accuracy from these checks
        self.val_loss_best.reset()

    def step(self, batch: Any, registration: bool = False):
        output = self.forward(batch[0], batch[1], registration=registration)
        loss_dict = self.loss(batch, output, int_downsize=self.net.int_downsize, registration=registration)

        return loss_dict, output

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict, output = self.step(batch, registration=False)

        # update and log metrics
        self.train_loss(loss_dict['loss'])
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.train_ssim_loss(loss_dict['ssim_loss'])
        # self.log("train/ssim_loss", self.train_ssim_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_smooth_loss(loss_dict['smoothness_loss'])
        self.log("train/smooth_loss", self.train_smooth_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_mse_dsa(batch[1], batch[0])
        self.log("train/mse_dsa", self.train_mse_dsa, on_step=False, on_epoch=True, prog_bar=True)
        self.train_mse_dla_warp_mask(batch[1], output[0])
        self.log("train/mse_dla_warp_mask", self.train_mse_dla_warp_mask, on_step=False, on_epoch=True, prog_bar=True)
        self.train_mse_dla_warp_contrast(batch[0], output[2])
        self.log("train/mse_dla_warp_contrast", self.train_mse_dla_warp_contrast, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return loss_dict

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict, output = self.step(batch, registration=False)

        # update and log metrics
        self.val_loss(loss_dict['loss'])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.val_ssim_loss(loss_dict['ssim_loss'])
        # self.log("val/ssim_loss", self.val_ssim_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_smooth_loss(loss_dict['smoothness_loss'])
        self.log("val/smooth_loss", self.val_smooth_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_mse_dsa(batch[1], batch[0])
        self.log("val/mse_dsa", self.val_mse_dsa, on_step=False, on_epoch=True, prog_bar=True)
        self.val_mse_dla_warp_mask(batch[1], output[0])
        self.log("val/mse_dla_warp_mask", self.val_mse_dla_warp_mask, on_step=False, on_epoch=True, prog_bar=True)
        self.val_mse_dla_warp_contrast(batch[0], output[2])
        self.log("val/mse_dla_warp_contrast", self.val_mse_dla_warp_contrast, on_step=False, on_epoch=True, prog_bar=True)

        return loss_dict

    def validation_epoch_end(self, outputs: List[Any]):
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        series, series_fname = batch[0].squeeze(), batch[1][0]  # assuming batch size = 1
        warped_series = torch.zeros_like(series)
        neg_warped_grids = torch.zeros_like(series)
        pos_warped_grids = torch.zeros_like(series)
        dsa, dla = torch.zeros_like(series), torch.zeros_like(series)
        warped_masks = torch.zeros_like(series)
        neg_flows = torch.zeros((series.shape[0], 2, series.shape[1], series.shape[2]), dtype=torch.double).cuda()
        pos_flows = torch.zeros((series.shape[0], 2, series.shape[1], series.shape[2]), dtype=torch.double).cuda()
        grid = torch.as_tensor(ndutils.bw_grid((series.shape[-2], series.shape[-1]), 10), dtype=torch.double).cuda()
        neg_flow_mags = torch.zeros(series.shape[0], dtype=torch.double).cuda()
        pos_flow_mags = torch.zeros(series.shape[0], dtype=torch.double).cuda()
        series_proc_time = 0
        for fnum in range(series.shape[0]):
            a = series[0].unsqueeze(0).unsqueeze(0)
            b = series[fnum].unsqueeze(0).unsqueeze(0)
            frame_proc_start = time.perf_counter()
            warped_mask, pos_flow, warped_contrast_frame, neg_flow = \
                self.forward(a, b,
                             registration=True)
            frame_proc_end = time.perf_counter()
            self.frame_proc_times.append(frame_proc_end-frame_proc_start)
            series_proc_time = series_proc_time + (frame_proc_end-frame_proc_start)
            warped_series[fnum] = warped_contrast_frame.squeeze()
            neg_flows[fnum], pos_flows[fnum] = neg_flow.double(), pos_flow.double()

            neg_flow_field = neg_flows[fnum].flip(dims=(0,)).field().from_pixels()
            neg_warped_grids[fnum] = neg_flow_field(grid.clone()).squeeze()
            neg_flow_mags[fnum] = neg_flow_field.pixels().magnitude().max()
            pos_flow_field = pos_flows[fnum].flip(dims=(0,)).field().from_pixels()
            pos_warped_grids[fnum] = pos_flow_field(grid.clone()).squeeze()
            pos_flow_mags[fnum] = pos_flow_field.pixels().magnitude().max()
            dsa[fnum] = series[fnum] - series[0]
            dla[fnum] = warped_series[fnum] - series[0]
            warped_masks[fnum] = warped_mask.squeeze()

            self.test_mse_dsa(series[0], series[fnum])
            self.test_mse_dla_warp_mask(warped_masks[fnum], series[fnum])
            self.test_mse_dla_warp_contrast(series[0], warped_series[fnum])

        self.series_proc_times.append(series_proc_time)
        self.log("test/mse_dsa", self.test_mse_dsa, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mse_dla_warp_mask", self.test_mse_dla_warp_mask, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mse_dla_warp_contrast", self.test_mse_dla_warp_contrast, on_step=False, on_epoch=True,
                 prog_bar=True)

        # mask_warped_dla_dir = os.path.join(self.trainer.log_dir, "vis", "mask_warped_dla")
        # Path(mask_warped_dla_dir).mkdir(parents=True, exist_ok=True)
        # contrast_warped_dla_dir = os.path.join(self.trainer.log_dir, "vis", "contrast_warped_dla")
        # Path(contrast_warped_dla_dir).mkdir(parents=True, exist_ok=True)
        # for fnum in range(series.shape[0]):
        #     contrast_fname = "{}_frame_{}".format(series_fname, fnum)
        #     np.save(os.path.join(mask_warped_dla_dir, f'{contrast_fname}.npy'), (series[fnum]-warped_masks[fnum]).cpu().numpy())
        #     np.save(os.path.join(contrast_warped_dla_dir, f'{contrast_fname}.npy'), (warped_series[fnum]-series[0]).cpu().numpy())
        
        print("Overall series processing time (len: {}): avg+std = {:.3f}\u00B1{:.2e}".format(len(self.series_proc_times), np.mean(self.series_proc_times), np.std(self.series_proc_times)))
        print("Overall frame  processing time (len: {}): avg+std = {:.3f}\u00B1{:.2e}".format(len(self.frame_proc_times), np.mean(self.frame_proc_times), np.std(self.frame_proc_times)))
        # series, dsa, warped_series, dla, neg_warped_grids = \
        #     (x.cpu() for x in (series, dsa, warped_series, dla, neg_warped_grids))
        # series_vis_dir = os.path.join(self.trainer.log_dir, "vis", "series")
        # Path(series_vis_dir).mkdir(parents=True, exist_ok=True)
        # visualize_series(series, series_fname, dsa,
        #                  warped_series, dla, neg_warped_grids, neg_flow_mags, save_dir=series_vis_dir)

        # warped_masks, pos_warped_grids = warped_masks.cpu(), pos_warped_grids.cpu()
        # pair_vis_dir = os.path.join(self.trainer.log_dir, "vis", "pairs")
        # Path(pair_vis_dir).mkdir(parents=True, exist_ok=True)
        # for fnum in range(series.shape[0]):
        #     contrast_fname = "{}_frame_{}".format(series_fname, fnum)
        #     visualize_pair(series[0], series[fnum], contrast_fname, warped_masks[fnum], pos_warped_grids[fnum],
        #                    pos_flow_mags[fnum], pair_vis_dir, warp_mask=True)
        #     visualize_pair(series[0], series[fnum], contrast_fname, warped_series[fnum], neg_warped_grids[fnum],
        #                    neg_flow_mags[fnum], pair_vis_dir, warp_mask=False)

    def test_epoch_end(self, outputs: List[Any]):
        print("Overall series processing time (len: {}): avg+std = {:.3f}\u00B1{:.2e}".format(len(self.series_proc_times), np.mean(self.series_proc_times), np.std(self.series_proc_times)))
        print("Overall frame  processing time (len: {}): avg+std = {:.3f}\u00B1{:.2e}".format(len(self.frame_proc_times), np.mean(self.frame_proc_times), np.std(self.frame_proc_times)))
        # pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
