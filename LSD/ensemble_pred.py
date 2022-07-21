"""
@author: Siyang Li
@date: Jun.13 2022
@email: lsyyoungll@gmail.com

ABAW4 Challenge
Learning from Synthetic data
Ensemble method: combining predictions from multiple models

Please modify dataset subfolder name ANGERE to ANGER.
"""

import os
import sys
import random
import time

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings
import torchvision.transforms as transforms

from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import Dataset, DataLoader
from skimage import io
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from datetime import date
from tqdm import tqdm
# from torchsummary import summary
from torchvision.models import resnet50, inception_v3, googlenet, vit_b_16, efficientnet_b0

def pred_ensemble(model_list, eids, dataset_name, ensemble_id):

    preds_all = []
    img_path_all = []

    id_str = ''

    for i in range(len(model_list)):
        model_name = model_list[i]
        eid = eids[i]
        id_str = id_str + eid
        predictions = np.load('./runs/predictions/predictions_' + dataset_name + eid +  '_' + model_name + '.npy')
        img_paths = np.load('./runs/predictions/file_names_' + dataset_name + eid + '_' + model_name + '.npy')
        preds_all.append(predictions)
        img_path_all.append(img_paths)

    preds_all = np.stack(preds_all)
    preds = np.sum(preds_all, 0)
    predicted = np.argmax(preds, 1)


    with open('./runs/result/ensemble' + ensemble_id + dataset_name + id_str + '.txt', 'w') as out:
        out.write('image,expression\n')
        for i in range(len(predicted)):
            out.write(img_path_all[0][i] + ',' + str(predicted[i]))
            out.write('\n')


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    num_classes = 6
    batch_size = 32
    num_workers = 0

    device_id = 7  # cuda device ID

    device = torch.device('cuda:' + str(device_id))

    ensemble_id = '_0_'
    experiment_time = str(date.today())

    print("Current date:", experiment_time)
    print('using cuda device ID:', device)

    model_list = ['vit_DLN', 'resnet_DLN', 'effnet_DLN', 'googlenet_DLN']
    eids = ['_720', '_710', '_730', '_740']

    #model_list = ['vit_DLN', 'resnet_DLN', 'effnet_DLN', 'googlenet_DLN']
    #eids = ['_810', '_820', '_830', '_840']

    #model_list = ['vit_DLN', 'resnet_DLN', 'effnet_DLN', 'googlenet_DLN', 'vit_DLN', 'resnet_DLN', 'effnet_DLN', 'googlenet_DLN']
    #eids = ['_720', '_710', '_730', '_740', '_810', '_820', '_830', '_840']

    print('Ensemble Models:', model_list)

    '''
    print('#' * 20)
    print('ensembling on AffectNet valid data...')
    dataset_name = 'AffectNet'
    pred_ensemble(model_list, eids, dataset_name, ensemble_id)
    
    print('#' * 20)
    print('ensembling on validation set data...')
    dataset_name = 'Synthetic_valid'
    pred_ensemble(model_list, eids, dataset_name, ensemble_id)
    '''

    print('#' * 20)
    print('ensembling on test set data...')
    dataset_name = 'Synthetic_test'
    pred_ensemble(model_list, eids, dataset_name, ensemble_id)



