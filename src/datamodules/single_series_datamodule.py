from os.path import splitext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeVar

import albumentations as A
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from albumentations import pytorch as AT
from pytorch_lightning import LightningDataModule
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

from src.datamodules.dicom_reader import reader as dicom_reader

T = TypeVar('T')
T_co = TypeVar("T_co", covariant=True)


class UnsubMRCLEANDataset(Dataset[T_co]):
    def __init__(self,
                 fp: str = 'data/',
                 frame_num: int = None,
                 image_size: int = 1024,
                 sample_as_series: bool = True,
                 transforms=None):
        self.transforms = transforms
        self.fp = fp
        self.frame_num = frame_num
        self.dst_img_size = image_size
        self.sample_as_series = sample_as_series
        self.series = self.load(self.fp)

        # self.series_paths = sorted(glob.glob(self.fp + "/R" + '[0-9]' * 4 + '/*.dcm', recursive=False))
        # assert self.series_paths, f'No image found in {self.fp}, make sure you specify the correct path.'
        # self.patient_ids = sorted(set([os.path.basename(os.path.dirname(f)) for f in self.series_paths]))
        # print(f'Dataset contains {len(self.patient_ids)} patients with {len(self.series_paths)} series.')

    def __len__(self):
        return 1 if (self.sample_as_series or self.frame_num is not None) else self.series.shape[0]

    @classmethod
    def preprocess(cls, img, dst_img_size, transforms=None):
        if img.ndim == 2:
            img = resize(img, (dst_img_size, dst_img_size), anti_aliasing=False, preserve_range=True)
        else:
            # TODO: double check the dimension order if 3D
            img = resize(img, (img.shape[0], dst_img_size, dst_img_size), anti_aliasing=False, preserve_range=True)

        if transforms is not None:
            img = transforms(image=img)['image']
        return img

    @classmethod
    def preprocess_series(cls, series, dst_img_size, transforms=None):
        series_tensor = torch.zeros((series.shape[0], 1, dst_img_size, dst_img_size))
        for fnum in range(series.shape[0]):
            frame = resize(series[fnum, :, :], (dst_img_size, dst_img_size), anti_aliasing=False, preserve_range=True)
            series_tensor[fnum, 0, :, :] = transforms(image=frame)['image']

        return series_tensor

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return np.load(filename)
        elif ext in ['.pt', '.pth']:
            return torch.load(filename).numpy()
        elif ext in ['.dcm']:
            ds = pydicom.dcmread(filename, defer_size="1 KB", stop_before_pixels=False, force=True)
            assert 2 ** (ds.BitsStored - 1) < ds.pixel_array.max() < 2 ** ds.BitsStored, \
                "Error: bits stored: {}, pixel value max: {}".format(ds.BitsStored, ds.pixel_array.max())
            return ds.pixel_array.astype(np.float32) / (2 ** ds.BitsStored - 1)  # produces float array in [0, 1] range.
        elif ext in ['.nii']:
            img_obj = nib.load(filename)
            return np.transpose(img_obj.get_fdata(), (1, 0, 2))
        else:
            return np.asarray(Image.open(filename))

    def __getitem__(self, idx):
        if self.sample_as_series:
            series = self.preprocess_series(self.series, self.dst_img_size, transforms=self.transforms)
            series_name = "{}_{}".format(Path(self.fp).parent.name, Path(self.fp).stem)
            return series, series_name
        if self.frame_num is not None:
            idx = self.frame_num
        mask = self.series[0, :, :]
        contrast_frame = self.series[idx, :, :]

        mask = self.preprocess(mask, self.dst_img_size, transforms=self.transforms)
        contrast_frame = self.preprocess(contrast_frame, self.dst_img_size, transforms=self.transforms)
        contrast_frame_name = "{}_{}_frame_{}".format(Path(self.fp).parent.name, Path(self.fp).stem, idx)
        return mask, contrast_frame, contrast_frame_name


class UnsubtractedMRCLEANDataModule(LightningDataModule):
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

    def __init__(
            self,
            fp: str = "data/unsub_mrclean/",
            frame_num: int = None,
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

        # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        # )

        self.train_transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5,
                               border_mode=1),  # cv2.BORDER_REPLICATE
            # A.Normalize(mean=0, std=1),
            AT.ToTensorV2(),
        ])

        self.transforms = A.Compose([
            # A.Normalize(mean=0, std=1),
            AT.ToTensorV2(),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if not self.data_train and not self.data_val and not self.data_test:
            train_transforms = self.train_transforms if self.hparams.augmentation else self.transforms
            self.data_train = UnsubMRCLEANDataset(self.hparams.fp,
                                                  frame_num=self.hparams.frame_num,
                                                  image_size=self.hparams.image_size,
                                                  transforms=train_transforms,
                                                  sample_as_series=self.hparams.sample_as_series[0])
            self.data_val = UnsubMRCLEANDataset(self.hparams.fp,
                                                frame_num=self.hparams.frame_num,
                                                image_size=self.hparams.image_size,
                                                transforms=self.transforms,
                                                sample_as_series=self.hparams.sample_as_series[1])
            self.data_test = UnsubMRCLEANDataset(self.hparams.fp,  # using val here, keeping test for the end.
                                                 frame_num=self.hparams.frame_num,
                                                 image_size=self.hparams.image_size,
                                                 transforms=self.transforms,
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
            batch_size=1,
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
