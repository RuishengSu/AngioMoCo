import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from torchmetrics import Metric, StructuralSimilarityIndexMeasure
from torch import Tensor, tensor
from typing import Any


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    def loss(self, preds, target, mask=None):
        """
        Mean squared error loss, with optional mask.
        Args:
            preds: Predictions from model
            target: Ground truth values
            mask: binary mask which tells which pixels should be counted for MSE calculation.
            positions with one will be used, while positions with zeros will be excluded.
        """
        if mask is None:
            mask = torch.ones_like(preds).bool()
        if preds.shape != target.shape or preds.shape != mask.shape:
            raise RuntimeError(
                f"Predictions, targets, and mask are expected to have the same shape, "
                f"but got {preds.shape}, {target.shape} and {mask.shape}."
            )
        diff = preds - target
        sum_squared_error = torch.sum(diff * diff * mask)
        n_obs = torch.sum(mask)

        return sum_squared_error / n_obs


class MaskedMSE(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_squared_error: Tensor
    total: Tensor

    def __init__(
            self,
            squared: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        self.squared = squared

    def update(self, preds: Tensor, target: Tensor, mask: Tensor = None) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
            mask: binary mask which tells which pixels should be counted for MSE calculation.
            positions with one will be used, while positions with zeros will be excluded.
        """
        if mask is None:
            mask = torch.ones_like(preds).bool()
        if preds.shape != target.shape or preds.shape != mask.shape:
            raise RuntimeError(
                f"Predictions, targets, and mask are expected to have the same shape, "
                f"but got {preds.shape}, {target.shape} and {mask.shape}."
            )
        diff = preds - target
        sum_squared_error = torch.sum(diff * diff * mask)
        n_obs = torch.sum(mask)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return self.sum_squared_error / self.total if self.squared else torch.sqrt(self.sum_squared_error / self.total)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1'):
        self.penalty = penalty

    def loss(self, y_pred, loss_mult=None):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        # dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            # dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy)  # + torch.mean(dz)
        grad = d / 2.0

        if loss_mult is not None:
            grad *= loss_mult
        return grad


# The classes below is adapted from
# https://github.com/jasonbian97/VoxelMorph-pl/blob/dd76e06dce1beb39cba3dc12fd4ad58b89d932df/src/lib/losses.py
class VMLoss(nn.Module):
    def __init__(self, lambda_param, image_loss='mse', bidir=False, mask_contrast=False, mask_edge=False):
        super().__init__()
        self.mask_contrast = mask_contrast
        self.mask_edge = mask_edge
        self.lambda_param = lambda_param
        self.bidir = bidir

        # prepare image loss
        # if image_loss == 'ncc':
        #     self.image_loss_func = NCC(win=NCC_win).loss
        if image_loss == 'mse':
            self.image_loss_func = MSE().loss
        elif image_loss == 'ssim':
            self.image_loss_func = StructuralSimilarityIndexMeasure()
        else:
            raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % image_loss)
        self.smooth_loss_func = Grad('l2').loss

    def forward(self, model_in, model_out, int_downsize=None, registration=False):
        """
        model_in: batch in put of the voxelmorph model, [0]: mask frame, [1]: contrast_frame.
        model_out: output of the voxelmorph model,
                    when in registration mode, [0-2] are warped mask, preint-flow, and warped_contrast;
                    when not in registration mode, [0-3] are warped mask, pos_flow, warped_contrast, and neg_flow.
        registration: bool. The output struct of voxelmporh is different. We use this flag to avoid mis-use of outputs.
        """
        warped_mask, flow = model_out[0], model_out[1]
        mask_frame, contrast_frame = model_in[0], model_in[1]

        mw = int(mask_frame.shape[-1] / 16)  # loss mask width
        edge_mask = torch.ones_like(mask_frame).bool()
        if self.mask_edge:
            edge_mask = torch.zeros_like(mask_frame).bool()
            edge_mask[:, :, mw:-mw, mw:-mw] = True

        # pos direction similarity
        contrast_mask = torch.ones_like(mask_frame).bool()
        if self.mask_contrast:
            # contrast_mask = warped_x - y < 0
            contrast_mask = mask_frame - contrast_frame < 0.05
        loss_mask = torch.logical_and(edge_mask, contrast_mask)
        pos_similarity_loss = self.image_loss_func(warped_mask, contrast_frame, mask=loss_mask)
        lossd = {"pos_similarity_loss": pos_similarity_loss}  # for log purpose

        # neg direction similarity
        warped_contrast_frame = model_out[2]
        contrast_mask = torch.ones_like(mask_frame).bool()
        if self.mask_contrast:
            # contrast_mask = warped_x - y < 0
            contrast_mask = mask_frame - warped_contrast_frame < 0.05
        loss_mask = torch.logical_and(edge_mask, contrast_mask)
        neg_similarity_loss = self.image_loss_func(warped_contrast_frame, mask_frame, mask=loss_mask)
        lossd["neg_similarity_loss"] = neg_similarity_loss  # for log purpose

        # velocity smoothness
        if not registration:
            mw = int(mw/int_downsize)
            flow = flow[:, :, mw:-mw, mw:-mw] if self.mask_edge else flow
            lossd['smoothness_loss'] = self.smooth_loss_func(flow, loss_mult=int_downsize)  # for log purpose

        # total loss
        lossd['loss'] = pos_similarity_loss
        if self.bidir:
            lossd['loss'] = 0.5 * pos_similarity_loss + 0.5 * neg_similarity_loss
        if (not registration) and (self.lambda_param != 0):
            lossd['loss'] = (1 - self.lambda_param) * lossd['loss'] + self.lambda_param * lossd['smoothness_loss']
        return lossd
