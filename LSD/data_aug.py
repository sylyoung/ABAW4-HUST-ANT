"""
@author: Siyang Li
@date: Jun.13 2022
@email: lsyyoungll@gmail.com

ABAW4 Challenge
Learning from Synthetic data
Data Augmentation for Imbalanced Distribution

Please modify dataset subfolder name ANGERE to ANGER.
"""

import os
import sys
import random
import time

import pandas
import torch

import pandas as pd
import numpy as np
import warnings
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from skimage import io, img_as_ubyte
from matplotlib import pyplot as plt

from tqdm import tqdm


def create_EXPR_csv(data_dir, shuffle=False, split_percent=None, train_csv_path=None, valid_csv_path=None):
    def sort_func(name_string):
        if 'DS_Store' in name_string:
            return -1
        if name_string.endswith('.jpg'):
            name_string = name_string[:-4]
        while name_string.startswith('0') and len(name_string) != 1:
            name_string = name_string[1:]
        return int(name_string)

    class_names = ['ANGER', 'DISGUST', 'FEAR', 'HAPPINESS', 'SADNESS', 'SURPRISE']

    image_paths = []
    values = []

    for subdir, dirs, files in os.walk(data_dir):
        for file in sorted(files, key=sort_func):
            f_path = os.path.join(subdir, file)
            class_id = None
            for class_name in class_names:
                if class_name in f_path:
                    class_id = class_names.index(class_name)
                    break
            if f_path.endswith('.jpg'):
                if class_id is None:
                    print('ERROR!')
                    sys.exit(0)
                print(f_path, class_id)
                image_paths.append(f_path)
                values.append(class_id)

    image_paths = np.array(image_paths).reshape(-1, 1)
    values = np.array(values).reshape(-1, 1)

    if split_percent is not None:
        concat_train = []
        concat_valid = []
        for i in range(len(class_names)):
            inds = np.where(values == i)[0]
            image_paths_class = image_paths[inds]
            values_class = values[inds]
            concat_train.append(np.concatenate([image_paths_class[:int(image_paths_class.shape[0] * split_percent)],
                                                values_class[:int(values_class.shape[0] * split_percent)]], axis=1))
            concat_valid.append(np.concatenate([image_paths_class[int(image_paths_class.shape[0] * split_percent):],
                                                values_class[int(values_class.shape[0] * split_percent):]], axis=1))

        concat_train = np.concatenate(concat_train, axis=0)
        concat_valid = np.concatenate(concat_valid, axis=0)
        print(concat_train.shape)
        df1 = pd.DataFrame(concat_train)
        df1.to_csv(train_csv_path, index=False, header=False)
        print(concat_valid.shape)
        df2 = pd.DataFrame(concat_valid)
        df2.to_csv(valid_csv_path, index=False, header=False)
    else:
        concat_arr = np.concatenate([image_paths, values], axis=1)
        if shuffle:
            np.random.shuffle(concat_arr)
        print(concat_arr.shape)
        df = pd.DataFrame(concat_arr)
        df.to_csv(train_csv_path, index=False, header=False)


class CustomImagesDataset(Dataset):

    def __init__(self, csv_file, transform=None, device=None, no_aug_labels=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file, sep=',')
        self.device = device
        self.transform = transform
        self.no_aug_labels = no_aug_labels

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.csv_file.iloc[idx, 0]
        #print(img_path)

        image = io.imread(img_path)

        # apply transform
        if self.transform:
            image = self.transform(image).float()

        label = int(self.csv_file.iloc[idx, 1])
        #label = torch.Tensor([label]).to(torch.long)

        if label in self.no_aug_labels:
            return image, -1

        return image, label


def face_sentiment(device, batch_size=None, csv_path_train=None, csv_path_aug=None, aug_data_path=None,
                          transform=None, no_aug_labels=None):

    class_names = ['ANGER', 'DISGUST', 'FEAR', 'HAPPINESS', 'SADNESS', 'SURPRISE']

    train_dataset = CustomImagesDataset(csv_path_train, transform=transform, device=device, no_aug_labels=no_aug_labels)

    train_iterator = DataLoader(train_dataset, batch_size=batch_size)

    cnt = 0

    for batch in tqdm(train_iterator):
        data, label = batch

        data, label = data[0], label[0]
        data = data.permute(1, 2, 0).numpy()

        #io.imshow(data)
        #plt.show()
        label = int(label.numpy())
        if label == -1:
            continue
        file_path = aug_data_path + '/' + class_names[label] + '/' + str(cnt) + '.jpg'

        while os.path.isfile(file_path):
            cnt += 1
            file_path = aug_data_path + '/' + class_names[label] + '/' + str(cnt) + '.jpg'

        #io.imsave(file_path, data)
        #print(data)
        io.imsave(file_path, img_as_ubyte(data))
        df = pandas.DataFrame([[file_path, label]])
        df.to_csv(csv_path_aug, mode='a', index=False, header=False)


class ImagesDatasetAug(Dataset):

    def __init__(self, csv_file, transform=None, device=None, aug_label=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file, sep=',')
        self.device = device
        self.transform = transform
        self.aug_label = aug_label

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = int(self.csv_file.iloc[idx, 1])
        if label != self.aug_label:
            return -1, -1

        img_path = self.csv_file.iloc[idx, 0]
        image = io.imread(img_path)
        # apply transform
        if self.transform:
            image = self.transform(image).float()

        return image, label



def generate_aug(device, batch_size=None, csv_path_train=None, csv_path_aug=None, aug_data_path=None,
                          transform=None, aug_label=None, cnt=None, stop_ind=None):

    class_names = ['ANGER', 'DISGUST', 'FEAR', 'HAPPINESS', 'SADNESS', 'SURPRISE']

    aug_label = class_names.index(aug_label)

    train_dataset = ImagesDatasetAug(csv_path_train, transform=transform, device=device, aug_label=aug_label)

    train_iterator = DataLoader(train_dataset, batch_size=batch_size)

    for batch in tqdm(train_iterator):
        data, label = batch

        data, label = data[0], label[0]
        label = int(label.numpy())
        if label == -1:
            continue
        data = data.permute(1, 2, 0).numpy()

        file_path = aug_data_path + '/' + class_names[label] + '/' + str(cnt) + '.jpg'

        while os.path.isfile(file_path):
            cnt += 1
            file_path = aug_data_path + '/' + class_names[label] + '/' + str(cnt) + '.jpg'

        #io.imsave(file_path, data)
        #print(data)
        io.imsave(file_path, img_as_ubyte(data))
        df = pandas.DataFrame([[file_path, label]])
        df.to_csv(csv_path_aug, mode='a', index=False, header=False)

        if cnt == stop_ind:
            return cnt
    return cnt



if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    device = torch.device('cpu')

    batch_size = 1

    EXPR_train_data_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/training_set_synthetic_images'
    #EXPR_train_data_path = '/Users/Riccardo/Workspace/HUST-BCI/data/ABAW4/ABAW4-Synthetic/training_set_synthetic_images'
    #EXPR_train_data_path = '/Users/Riccardo/Workspace/HUST-BCI/data/AffectNet/train_set/images'

    EXPR_train_CSV_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/train.csv'
    #EXPR_train_CSV_path = '/Users/Riccardo/Workspace/HUST-BCI/data/ABAW4/ABAW4-Synthetic/train_local.csv'
    #EXPR_train_CSV_path = '/Users/Riccardo/Workspace/HUST-BCI/data/AffectNet/train_set/train.csv'

    EXPR_aug_data_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/aug'
    #EXPR_aug_data_path = '/Users/Riccardo/Workspace/HUST-BCI/data/ABAW4/ABAW4-Synthetic/randaug'
    #EXPR_aug_data_path = '/Users/Riccardo/Workspace/HUST-BCI/data/AffectNet/train_set/randaug'

    EXPR_aug_CSV_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/aug.csv'
    #EXPR_aug_CSV_path = '/Users/Riccardo/Workspace/HUST-BCI/data/ABAW4/ABAW4-Synthetic/randaug.csv'
    #EXPR_aug_CSV_path = '/Users/Riccardo/Workspace/HUST-BCI/data/AffectNet/train_set/randaug.csv'


    if not os.path.isdir(EXPR_aug_data_path):
        os.mkdir(EXPR_aug_data_path)
    class_names = ['ANGER', 'DISGUST', 'FEAR', 'HAPPINESS', 'SADNESS', 'SURPRISE']
    for class_name in class_names:
        class_dir = EXPR_aug_data_path + '/' + class_name
        if not os.path.isdir(class_dir):
            os.mkdir(class_dir)


    randaug = transforms.Compose(
        [transforms.ToTensor(),
         transforms.ConvertImageDtype(torch.uint8),
         transforms.RandAugment(num_ops=2, magnitude=9),
         transforms.ConvertImageDtype(torch.float32)])

    #aug_count = [144631 - 18286, 144631 - 15150, 144631 - 10923, 144631 - 73285, 0, 144631 - 14976]
    aug_count = [134415 - 24882, 134415 - 3803, 134415 - 6378, 0, 134415 - 25459, 134415 - 14090]

    for stop_ind in aug_count:
        curr_cnt = 0
        cls = class_names[aug_count.index(stop_ind)]
        print(cls)
        while curr_cnt < stop_ind:
            curr_cnt = generate_aug(device, batch_size, EXPR_train_CSV_path, EXPR_aug_CSV_path, EXPR_aug_data_path,
                         randaug, cls, curr_cnt, stop_ind)
            print(curr_cnt)

