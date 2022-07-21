import os
from PIL import Image, UnidentifiedImageError
import pandas as pd
import torch
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils import data as data

from data_utils import train_transforms, test_transforms

class DatasetBase(data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _get_all_label(self):
        raise NotImplementedError


class ImageDataset(DatasetBase):
    def __init__(self, data_transforms, data_dir_path=None, annotation_file=None):
        self._transform = data_transforms
        self.annotation_file = annotation_file
        self.data_dir_path = data_dir_path
        self.df = pd.read_csv(os.path.join(data_dir_path, annotation_file))

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        img_path = os.path.join(self.data_dir_path, row[0])
        try:
            image = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            image = Image.open(os.path.join(self.data_dir_path+'_unaligned', row[0])).convert("RGB")
            print('cannot find aligned image {}'.format(row[0]))
        image = self._transform(image)
        if len(row) > 1 and np.sum(row[1:].map(lambda x: np.isnan(x)))==0:
            label = row[1:].values.astype(float)
            return image, label
        else:
            return image

    def _get_all_label(self):
        if self.df.shape[1] > 1 and not np.isnan(self.df.iloc[0, 1]):
            return self.df.iloc[:, 1:].values

    @property
    def ids(self):
        return np.arange(len(self.df))

    @property
    def dataset_size(self):
        return len(self.ids)

    def __len__(self):
        return self.dataset_size


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = list(datasets)
        self.datasets_lengths = [len(d) for d in self.datasets]
        self.length = max(self.datasets_lengths)
        for i in range(len(self.datasets)):
            if len(self.datasets[i]) < self.length:
                N = len(self.datasets[i])
                dataset_df = self.datasets[i].df
                for _ in range(self.length // N):
                    self.datasets[i].df = pd.concat([self.datasets[i].df, dataset_df])
                print("dataset {} resampled from {} to {} images".format(i, N, len(self.datasets[i])))
        self.datasets_lengths = [len(d) for d in self.datasets]
        self.length = min(self.datasets_lengths)

    def __getitem__(self, i):
        # 对于样本少的dataset直接从头再取
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return self.length


class MTDataModule(pl.LightningDataModule):
    def __init__(self, train_set, val_set, batch_size, num_workers=4, *args, **kwargs):
        super(MTDataModule, self).__init__(*args, **kwargs)
        self.train_set = ConcatDataset(*train_set)
        self.val_set = val_set
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dl_train = data.DataLoader(self.train_set, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return dl_train

    def val_dataloader(self):
        val_batch_size = self.batch_size*3
        dl_val = [data.DataLoader(valset, num_workers=self.num_workers, batch_size=val_batch_size,
                                  shuffle=False, drop_last=False) for valset in self.val_set]
        return dl_val


def get_Dataset_TrainVal(data_dir_path: str, train_annotation_file:str, val_annotation_file:str, data_transforms=None):
    ds_train = ImageDataset(data_transforms, data_dir_path, train_annotation_file)
    ds_val = ImageDataset(data_transforms, data_dir_path, val_annotation_file)
    return ds_train, ds_val


def get_MTL_datamodule(img_size, batch_size, num_workers=4):
    dir_path='ABAW5-MTL'
    train_anno_file = "training_{}_annotation.csv"
    val_anno_file = "validation_{}_annotation.csv"
    ds_AU = get_Dataset_TrainVal(data_dir_path=dir_path, train_annotation_file=train_anno_file.format('AU'),
            val_annotation_file=val_anno_file.format('AU'), data_transforms=train_transforms(img_size))
    ds_EXPR = get_Dataset_TrainVal(data_dir_path=dir_path, train_annotation_file=train_anno_file.format('EXPR'),
            val_annotation_file=val_anno_file.format('EXPR'), data_transforms=train_transforms(img_size))
    ds_VA = get_Dataset_TrainVal(data_dir_path=dir_path, train_annotation_file=train_anno_file.format('VA'),
            val_annotation_file=val_anno_file.format('VA'), data_transforms=train_transforms(img_size))

    dm = MTDataModule([ds_AU[0], ds_EXPR[0], ds_VA[0]], [ds_AU[1], ds_EXPR[1], ds_VA[1]], batch_size, num_workers)
    return dm


if __name__ == '__main__':
    dm = get_MTL_datamodule(img_size=299, batch_size=16, num_workers=4)
    dl_train = dm.train_dataloader()
    for batch in dl_train:
        (data_au, label_au), (data_expr, label_expr), (data_va, label_va) = batch
        print(data_au.size())


def get_test_dataset(dir_path, test_sample_list):
    img_size = 299
    ds_test = ImageDataset(test_transforms(img_size), dir_path, test_sample_list)
    test_dataloader = data.DataLoader(ds_test, batch_size=32, shuffle=False,
                                                  num_workers=8, drop_last=False)
    return test_dataloader
