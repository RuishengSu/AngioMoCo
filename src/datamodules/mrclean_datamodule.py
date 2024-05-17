from glob import glob
import os
import random
from os.path import splitext
from typing import Any, Dict, Optional, Tuple, Sequence, Union, List, TypeVar
import re
import albumentations as A
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from albumentations import pytorch as AT
from pytorch_lightning import LightningDataModule
from skimage.transform import resize
from torch import randperm, Generator, default_generator
from torch._utils import _accumulate
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from src.datamodules.dicom_reader import reader as dicom_reader

T = TypeVar('T')
T_co = TypeVar("T_co", covariant=True)


class UnsubMRCLEANDataset(Dataset[T_co]):
    def __init__(self,
                 image_dir: str = 'data/',
                 patient_ids: List = None,
                 image_size: int = 1024,
                 sample_as_series: bool = True,
                 augmentation: bool = False):
        if patient_ids is None:
            print(
                "Warning: no patient_ids provided. Collecting all patients from the given image_dir.")
            self.file_paths = sorted(
                glob(os.path.join(image_dir, "R" + '[0-9]' * 4, '*.*'), recursive=False))
            self.patient_ids = sorted(
                set([os.path.basename(os.path.dirname(f)) for f in self.file_paths]))
        else:
            self.patient_ids = patient_ids
            self.file_paths = sorted(
                set(sum([glob(os.path.join(image_dir, p, '*.*')) for p in patient_ids], [])))

        self.frames = []
        for fp in self.file_paths:
            if ".dcm" in fp:
                ds = pydicom.dcmread(
                    fp, defer_size="1 KB", stop_before_pixels=True, force=True)
                num_of_frames = ds.NumberOfFrames
                self.frames.extend([{'patient_id': os.path.basename(os.path.dirname(f)),
                                     'series_path': fp, 'fnum': i} for i in range(num_of_frames)])
            elif ".nii" in fp:
                img_obj = nib.load(fp)
                # assuming last dim is time for nii format.
                num_of_frames = img_obj.header.get_data_shape()[-1]
                self.frames.extend([{'patient_id': os.path.basename(os.path.dirname(f)),
                                     'series_path': fp, 'fnum': i} for i in range(num_of_frames)])
            elif ".npy" in fp:
                fnum = int(re.search(
                    '{}([0-9]+){}'.format("_frame_", ".npy"), fp).group(1))
                fp = fp.split('_frame_')[0] + "*.npy"
                self.frames.append({'patient_id': os.path.basename(os.path.dirname(fp)),
                                    'series_path': fp, 'fnum': fnum})
        self.series_paths = list(set(d['series_path'] for d in self.frames))
        self.augmentation = augmentation
        self.image_dir = image_dir
        self.dst_img_size = image_size
        self.sample_as_series = sample_as_series

        self.base_transforms = A.Compose([
            # A.Normalize(mean=0, std=1),
            AT.ToTensorV2(), ],
            additional_targets={'image0': 'image'})
        self.aug_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5,
                               border_mode=1),  # cv2.BORDER_REPLICATE
            AT.ToTensorV2(), ],
            additional_targets={'image0': 'image'})
        print(f'Created subset with {len(self.patient_ids)} patients, '
              f'{len(self.series_paths)} series, {len(self.frames)} frames. Patient IDs: {self.patient_ids}')

    def __len__(self):
        return len(self.series_paths) if self.sample_as_series else len(self.frames)

    @classmethod
    def preprocess(cls, img, dst_img_size, transforms=None):
        if img.ndim == 2:
            img = resize(img, (dst_img_size, dst_img_size),
                         anti_aliasing=False, preserve_range=True)
        else:
            # double check the dimension order if 3D
            img = resize(img, (img.shape[0], dst_img_size, dst_img_size),
                         anti_aliasing=False, preserve_range=True)

        if transforms is not None:
            img = transforms(image=img)['image']
        return img

    @classmethod
    def preprocess_series(cls, series, dst_img_size, transforms=None):
        series_tensor = torch.zeros(
            (series.shape[0], 1, dst_img_size, dst_img_size))
        for fnum in range(series.shape[0]):
            frame = resize(series[fnum, :, :], (dst_img_size, dst_img_size),
                           anti_aliasing=False, preserve_range=True)
            series_tensor[fnum, 0, :, :] = transforms(image=frame)['image']

        return series_tensor

    @classmethod
    def load(cls, filename):
        if filename[-5:] == "*.npy":
            return np.stack([np.load(f) for f in sorted(glob(filename))], axis=0)
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return np.load(filename)
        if ext in ['.pt', '.pth']:
            return torch.load(filename).numpy()
        if ext in ['.dcm']:
            ds = pydicom.dcmread(
                filename, defer_size="1 KB", stop_before_pixels=False, force=True)
            assert 2 ** (ds.BitsStored - 1) < ds.pixel_array.max() < 2 ** ds.BitsStored, \
                "Error: bits stored: {}, pixel value max: {}".format(
                    ds.BitsStored, ds.pixel_array.max())
            # produces float array in [0, 1] range.
            return ds.pixel_array.astype(np.float32) / (2 ** ds.BitsStored - 1)
        if ext in ['.nii']:
            img_obj = nib.load(filename)
            # assuming value range [0, 1].
            return np.transpose(img_obj.get_fdata(), (2, 1, 0)).astype(np.float32)
        if os.path.isdir(filename):

            return np.asarray(Image.open(filename))

    def __getitem__(self, idx):
        if self.sample_as_series:
            series_path = self.series_paths[idx]
            series = self.load(series_path)

            series = self.preprocess_series(
                series, self.dst_img_size, transforms=self.base_transforms)
            series_name = "{}_{}".format(
                Path(series_path).parent.name, Path(series_path).stem.replace('*', ''))
            return series, series_name
        else:
            frame_dict = self.frames[idx]
            series_path, fnum = frame_dict['series_path'], frame_dict['fnum']
            if ".npy" in series_path:
                mask_frame = self.load(series_path.replace('*', '_frame_00'))
                contrast_frame = self.load(
                    series_path.replace('*', f'_frame_{fnum:02d}'))
            else:
                series = self.load(series_path)
                mask_frame = series[0, :, :]
                contrast_frame = series[fnum, :, :]

            mask_frame = self.preprocess(mask_frame, self.dst_img_size, transforms=self.transforms if self.augmentation else self.base_transforms)
            contrast_frame = self.preprocess(contrast_frame, self.dst_img_size, transforms=self.transforms if self.augmentation else self.base_transforms)
            contrast_frame_name = "{}_{}_frame_{}".format(
                Path(series_path).parent.name, Path(series_path).stem.replace('*', ''), fnum)
            return mask_frame, contrast_frame, contrast_frame_name


def patient_wise_random_split(data_dir: str, fractions: Sequence[Union[int, float]],
                              generator: Optional[Generator] = default_generator) -> List[List[str]]:
    r"""
    ===== Customized Patient-wise random split =====
    Args:
            Args:
        dataset (Dataset): Dataset to be split
        fractions (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    assert sum(fractions) == 1, "Invalid fractions!"

    series_paths = sorted(
        glob(os.path.join(data_dir, "R" + '[0-9]' * 4, '*.*'), recursive=False))
    assert series_paths, f'No image found in {data_dir}, make sure you specify the correct path.'
    patient_ids = sorted(
        set([os.path.basename(os.path.dirname(f)) for f in series_paths]))
    print(
        f'The whole train_val_test dataset contains {len(patient_ids)} patients with {len(series_paths)} series.')

    lengths = [int(len(patient_ids) * frac) for frac in fractions]
    lengths[0] = len(patient_ids) - sum(lengths[1:])

    patient_indices = randperm(len(patient_ids), generator=generator).tolist()
    patient_id_splits = []
    # series_index_splits = []
    for offset, length in zip(_accumulate(lengths), lengths):
        patient_id_subset = [patient_ids[i]
                             for i in patient_indices[offset - length: offset]]
        patient_id_splits.append(patient_id_subset)
        # series_indices = []
        # for p in patient_ids:
        #     series_indices.extend([i for i, elem in enumerate(series_paths) if p in elem])
        # series_index_splits.append(series_indices)
    # return [Subset(copy.copy(dataset), series_index_splits[i]) for i in range(len(fractions))]
    return patient_id_splits


class UnsubtractedMRCLEANDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/unsub_mrclean/",
            train_val_test_split: Tuple[float, float, float] = (0.5, 0.2, 0.3),
            sample_as_series: Tuple[bool, bool, bool] = (False, False, True),
            image_size: int = 1024,
            batch_size: int = 64,
            augmentation: bool = False,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    """LightningDataModule for MR CLEAN outcome prediction dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            patients_train, patients_val, patients_test = patient_wise_random_split(
                data_dir=self.hparams.data_dir,
                fractions=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(5),
            )
            self.data_train = UnsubMRCLEANDataset(self.hparams.data_dir,
                                                  patient_ids=patients_train,
                                                  image_size=self.hparams.image_size,
                                                  augmentation=self.hparams.augmentation,
                                                  sample_as_series=self.hparams.sample_as_series[0])
            self.data_val = UnsubMRCLEANDataset(self.hparams.data_dir,
                                                patient_ids=patients_val,
                                                image_size=self.hparams.image_size,
                                                augmentation=False,
                                                sample_as_series=self.hparams.sample_as_series[1])
            self.data_test = UnsubMRCLEANDataset(self.hparams.data_dir,
                                                 # TODO: Temporally using val for dev. Change to patients_test in the end. 
                                                 patient_ids=patients_val,
                                                 image_size=self.hparams.image_size,
                                                 augmentation=False,
                                                 sample_as_series=self.hparams.sample_as_series[2])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,  # if self.hparams.sample_as_series[2] else self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
