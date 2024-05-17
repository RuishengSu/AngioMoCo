import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils
import pydicom
import numpy as np
import albumentations as A
from albumentations import pytorch as AT
from skimage.transform import resize
from pathlib import Path
import torch
from src.models.components import ndutils
from src.models.components.utils_mrclean import visualize_series, visualize_pair
import os
import glob

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def load(cfg: DictConfig):
    """Evaluates given checkpoint on a single data sample.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    object_dict = {
        "cfg": cfg,
        "model": model,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    return model, logger


def prepare_image_pair(series_path, mov_fnum, img_size=512):
    ds = pydicom.dcmread(series_path, defer_size="1 KB", stop_before_pixels=False, force=True)
    assert 2 ** (ds.BitsStored - 1) < ds.pixel_array.max() < 2 ** ds.BitsStored
    series = ds.pixel_array.astype(np.float32) / (2 ** ds.BitsStored - 1)  # produces float array in [0, 1] range.
    transforms = A.Compose([AT.ToTensorV2()])

    mask = resize(series[0, :, :], (img_size, img_size), anti_aliasing=False, preserve_range=True)
    mask = transforms(image=mask)['image']

    contrast_frame = resize(series[mov_fnum, :, :], (img_size, img_size), anti_aliasing=False, preserve_range=True)
    contrast_frame = transforms(image=contrast_frame)['image']

    contrast_fname = "{}_{}_frame_{}".format(Path(series_path).parent.name, Path(series_path).stem, mov_fnum)

    return mask, contrast_frame, contrast_fname


def prepare_series(series_path, img_size=512):
    ds = pydicom.dcmread(series_path, defer_size="1 KB", stop_before_pixels=False, force=True)
    assert 2 ** (ds.BitsStored - 1) < ds.pixel_array.max() < 2 ** ds.BitsStored
    series = ds.pixel_array.astype(np.float32) / (2 ** ds.BitsStored - 1)  # produces float array in [0, 1] range.
    transforms = A.Compose([AT.ToTensorV2()])

    series_tensor = torch.zeros((series.shape[0], 1, img_size, img_size))
    for fnum in range(series.shape[0]):
        frame = resize(series[fnum, :, :], (img_size, img_size), anti_aliasing=False, preserve_range=True)
        series_tensor[fnum, :, :] = transforms(image=frame)['image']

    return series_tensor


def predict_series(model, fp, save_dir):
    series_tensor = prepare_series(fp)
    mask = series_tensor[0, :, :, :]
    warped_series = torch.zeros_like(series_tensor)
    warped_grid_series = torch.zeros_like(series_tensor)
    dsa = torch.zeros_like(series_tensor)
    dla = torch.zeros_like(series_tensor)
    flow_list = []
    for fnum in range(series_tensor.shape[0]):
        with torch.no_grad():
            _, _, warped_contrast_frame, neg_flow = model(mask.unsqueeze(0), series_tensor[fnum, :, :, :].unsqueeze(0), registration=True)
        grid = torch.from_numpy(ndutils.bw_grid(mask.squeeze().shape, 10)).double()
        neg_flow = neg_flow.double().flip(dims=(0,)).field().from_pixels().linverse()
        warped_series[fnum, :, :, :] = warped_contrast_frame.squeeze(0)
        warped_grid_series[fnum, :, :, :] = neg_flow(grid)

        dsa[fnum, :, :, :] = series_tensor[fnum, :, :, :] - mask
        dla[fnum, :, :, :] = warped_series[fnum, :, :, :] - mask
        flow_list.append(neg_flow)

    series_fname = "{}_{}".format(Path(fp).parent.name, Path(fp).stem)
    visualize_series(series_tensor.squeeze(), series_fname, dsa.squeeze(),
                     warped_series.squeeze(), dla.squeeze(), warped_grid_series.squeeze(), flow_list, save_dir=save_dir)


def predict_pairs(model, fp, mov_fnum, save_dir):
    series_tensor = prepare_series(fp)
    mov_fnums = []
    if isinstance(mov_fnum, int):
        mov_fnums = [mov_fnum]
    elif isinstance(mov_fnum, list):
        mov_fnums = mov_fnum
    elif isinstance(mov_fnum, str) and mov_fnum == 'all':
        mov_fnums = range(series_tensor.shape[0])
    for fnum in mov_fnums:
        mask, contrast_frame = series_tensor[0, :, :, :], series_tensor[fnum, :, :, :]
        contrast_fname = "{}_{}_{}".format(Path(fp).parent.name, Path(fp).stem, fnum)
        with torch.no_grad():
            warped_mask, pos_flow, warped_contrast_frame, neg_flow = model(mask.unsqueeze(0), contrast_frame.unsqueeze(0), registration=True)
        visualize_pair(mask, contrast_frame, contrast_fname, warped_mask, pos_flow, save_dir=save_dir)
        visualize_pair(mask, contrast_frame, "{}_inv".format(contrast_fname), warped_contrast_frame, neg_flow,
                       warp_mask=False, save_dir=save_dir)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    model, logger = load(cfg)
    checkpoint = torch.load(cfg.ckpt_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    log.info("Starting prediction!")
    if os.path.isfile(cfg.fp):
        if 'contrast_fnum' in cfg and (cfg.contrast_fnum is not None):
            predict_pairs(model, cfg.fp, cfg.contrast_fnum, cfg.paths.output_dir)
        else:
            predict_series(model, cfg.fp, cfg.paths.output_dir)
    elif os.path.isdir(cfg.fp):
        fp_list = sorted(glob.glob(os.path.join(cfg.fp, '**', '*.dcm'), recursive=True))
        for fp in fp_list:
            predict_series(model, fp, cfg.paths.output_dir)


if __name__ == "__main__":
    main()
