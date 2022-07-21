"""
@author: Siyang Li
@date: Jun.13 2022
@email: lsyyoungll@gmail.com

ABAW4 Challenge
Learning from Synthetic data
Model Testing Inference

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
import matplotlib.pyplot as plt

from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import Dataset, DataLoader
from skimage import io
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from torchvision.models import resnet50, inception_v3, googlenet, vit_b_16, efficientnet_b0


class CustomImagesDataset(Dataset):

    def __init__(self, csv_file, transform=None, device=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file, header=None, sep=',')
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.csv_file.iloc[idx, 0]

        image = io.imread(img_path)

        # apply transform
        if self.transform:
            image = self.transform(image).float()

        # convert label to tensor
        try:
            label = int(self.csv_file.iloc[idx, 1])
        except:
            # no labels
            return image, torch.Tensor([-1]).to(torch.long), img_path
        label = torch.Tensor([label]).to(torch.long)

        return image, label, img_path

class FC_classifier(nn.Module):

    def __init__(self, fc_num=0, out_chann=0):
        super(FC_classifier, self).__init__()

        # FC Layer
        self.fc = nn.Linear(fc_num, out_chann)

    def forward(self, x):
        x = self.fc(x)
        return x

def face_sentiment_resnet(device, num_classes=None, batch_size=None, num_workers=0, csv_path_test=None, model_arc=None, eid=None, no_label=False):

    pretrained_on_AffectNet = False

    if model_arc == 'vit':
        feature = vit_b_16(pretrained=True)
        del feature.heads
        feature.to(device)
        clf = FC_classifier(768, num_classes)
        for para in clf.parameters():
            para.requires_grad = True
        clf.to(device)
        model = [feature, clf]
    elif model_arc == 'effnet':
        model = efficientnet_b0(pretrained=True)
        model_dict = model.state_dict()
        new_state_dict = {}
        for k, v in model_dict.items():
            if not k.startswith('classifier.'):
                new_state_dict[k] = v
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        model.classifier = nn.Sequential(
            nn.Linear(1280, num_classes),
        )
        for para in model.classifier.parameters():
            para.requires_grad = True
        model.to(device)
    elif model_arc == 'resnet':
        model = resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        for para in model.fc.parameters():
            para.requires_grad = True
        model.to(device)
        #
    elif model_arc == 'googlenet':
        model = googlenet(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        for para in model.fc.parameters():
            para.requires_grad = True
        model.to(device)
        #
    elif model_arc == 'vit_DLN':
        model_identity = vit_b_16(pretrained=True)
        del model_identity.heads
        for para in model_identity.parameters():
            para.requires_grad = False
        model_face = vit_b_16(pretrained=True)
        del model_face.heads
        for para in model_face.parameters():
            para.requires_grad = True
        clf = FC_classifier(768, 6)
        model_identity.to(device)
        model_face.to(device)
        clf.to(device)
        model_face.load_state_dict(
            torch.load('./runs/baseline/model_face_best_' + '2022-07-21' + eid + '.pt'))
        clf.load_state_dict(
            torch.load('./runs/baseline/model_clf_best_' + '2022-07-21' + eid + '.pt'))
        # 810 load this too
        if pretrained_on_AffectNet:
            model_identity.load_state_dict(
                torch.load('./runs/baseline/model_feature_epoch_10_' + '2022-07-14' + '_126' + '.pt'))  # 810
        model = [model_identity, model_face, clf]
    elif model_arc == 'resnet_DLN':
        model_identity = resnet50(pretrained=True)
        for para in model_identity.parameters():
            para.requires_grad = False
        modules = list(model_identity.children())[:-1]  # delete the last fc layer.
        model_identity = nn.Sequential(*modules)
        model_face = resnet50(pretrained=True)
        for para in model_face.parameters():
            para.requires_grad = True
        modules = list(model_face.children())[:-1]  # delete the last fc layer.
        model_face = nn.Sequential(*modules)
        clf = FC_classifier(2048, 6)
        model_identity.to(device)
        model_face.to(device)
        clf.to(device)
        model_face.load_state_dict(
            torch.load('./runs/baseline/model_face_best_' + '2022-07-20' + eid + '.pt')) # 07-21 if 820
        clf.load_state_dict(
            torch.load('./runs/baseline/model_clf_best_' + '2022-07-20' + eid + '.pt')) # 07-21 if 820
        # 820 load this too
        if pretrained_on_AffectNet:
            loaded_state_dict = torch.load('./runs/baseline/model_epoch_10_' + '2022-07-14' + '_127' + '.pt')
            current_model_dict = model_identity.state_dict()
            new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
                              zip(current_model_dict.keys(), loaded_state_dict.values())}
            model_identity.load_state_dict(new_state_dict, strict=False)
        model = [model_identity, model_face, clf]
    elif model_arc == 'effnet_DLN':
        model_identity = efficientnet_b0(pretrained=True)
        for para in model_identity.parameters():
            para.requires_grad = False
        modules = list(model_identity.children())[:-1]  # delete the last fc layer.
        model_identity = nn.Sequential(*modules)
        model_face = efficientnet_b0(pretrained=True)
        for para in model_face.parameters():
            para.requires_grad = True
        modules = list(model_face.children())[:-1]  # delete the last fc layer.
        model_face = nn.Sequential(*modules)
        clf = FC_classifier(1280, 6)
        model_identity.to(device)
        model_face.to(device)
        clf.to(device)
        model_face.load_state_dict(
            torch.load('./runs/baseline/model_face_best_' + '2022-07-21' + eid + '.pt'))
        clf.load_state_dict(
            torch.load('./runs/baseline/model_clf_best_' + '2022-07-21' + eid + '.pt'))
        # 830 load this too
        if pretrained_on_AffectNet:
            loaded_state_dict = torch.load('./runs/baseline/model_epoch_10_' + '2022-07-14' + '_128' + '.pt')
            current_model_dict = model_identity.state_dict()
            new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
                              zip(current_model_dict.keys(), loaded_state_dict.values())}
            model_identity.load_state_dict(new_state_dict, strict=False)
        model = [model_identity, model_face, clf]
    elif model_arc == 'googlenet_DLN':
        model_identity = googlenet(pretrained=True)
        for para in model_identity.parameters():
            para.requires_grad = False
        modules = list(model_identity.children())[:-1]  # delete the last fc layer.
        model_identity = nn.Sequential(*modules)
        model_face = googlenet(pretrained=True)
        for para in model_face.parameters():
            para.requires_grad = True
        modules = list(model_face.children())[:-1]  # delete the last fc layer.
        model_face = nn.Sequential(*modules)
        clf = FC_classifier(1024, 6)
        model_identity.to(device)
        model_face.to(device)
        clf.to(device)
        model_face.load_state_dict(
            torch.load('./runs/baseline/model_face_best_' + '2022-07-21' + eid + '.pt'))
        clf.load_state_dict(
            torch.load('./runs/baseline/model_clf_best_' + '2022-07-21' + eid + '.pt'))
        # 840 load this too
        if pretrained_on_AffectNet:
            loaded_state_dict = torch.load('./runs/baseline/model_epoch_10_' + '2022-07-14' + '_129' + '.pt')
            current_model_dict = model_identity.state_dict()
            new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
                              zip(current_model_dict.keys(), loaded_state_dict.values())}
            model_identity.load_state_dict(new_state_dict, strict=False)
        model = [model_identity, model_face, clf]

    img_size = 224

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize([img_size, img_size]),
         #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]) # ImageNet
         transforms.Normalize(mean=(0.4734, 0.3544, 0.3267), std=(0.1891, 0.1597, 0.1504))]) # training set

    test_dataset = CustomImagesDataset(csv_path_test, transform=transform, device=device)
    test_iterator = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


    def evaluate(model, iterator, criterion):

        print('evaluating...')

        epoch_loss = 0

        model_identity, model_face, clf = model
        model_identity.eval()
        model_face.eval()
        clf.eval()

        predicted_all = []
        labels_all = []
        img_path_all = []
        predictions_all = []

        with torch.no_grad():
            for batch in tqdm(iterator):
                data, labels, img_path = batch
                data, labels = data.to(device), labels.to(device)

                out = model_face(data) - model_identity(data)

                while len(out.shape) > 2:
                    out = torch.squeeze(out, 2)

                predictions = clf(out)

                _, predicted = torch.max(predictions.cpu().data, 1)
                if not no_label:
                    loss = criterion(predictions, labels.reshape(-1, ))
                    epoch_loss += loss.item()

                predicted_all.append(predicted)
                labels_all.append(labels.detach().cpu())
                img_path_all.append(img_path)
                predictions_all.append(torch.softmax(predictions.cpu().data, dim=1))


        predicted_all = np.concatenate(predicted_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        img_path_all = np.concatenate(img_path_all, axis=0)
        predictions_all = np.concatenate(predictions_all, axis=0)

        predicted_all = predicted_all.reshape(-1, 1)
        labels_all = labels_all.reshape(-1, 1)

        dataset_name = None
        if 'AffectNet' in csv_path_test:
            dataset_name = 'AffectNet'
        elif 'Synthetic' in csv_path_test and 'valid' in csv_path_test:
            dataset_name = 'Synthetic_valid'
        elif 'Synthetic' in csv_path_test and 'test' in csv_path_test:
            dataset_name = 'Synthetic_test'
        print('dataset_name:', dataset_name)
        np.save('./runs/predictions/predictions_' + dataset_name + eid + '_' + model_arc, predictions_all)
        np.save('./runs/predictions/file_names_' + dataset_name + eid + '_' + model_arc, img_path_all)

        if not no_label:
            f1_arr = f1_score(labels_all, predicted_all, average=None)
            epoch_acc = accuracy_score(labels_all, predicted_all)
            conf_mat = confusion_matrix(labels_all, predicted_all)

            avg_loss = epoch_loss / len(iterator)

            return avg_loss, epoch_acc, f1_arr, conf_mat
        else:
            return

    criterion_eval = nn.CrossEntropyLoss()

    if not no_label:
        test_loss, test_acc, f1_arr, conf_mat = evaluate(model, test_iterator, criterion_eval)

        print(f'Test Loss/Accuracy/F1: {test_loss:.5f} {test_acc:.5f} {np.average(f1_arr):.5f}')
        print(f'Test F1: ', f1_arr)
        print('Confusion Matrix')
        print(conf_mat)
    else:
        evaluate(model, test_iterator, criterion_eval)



if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    num_classes = 6
    batch_size = 256
    num_workers = 8

    model_arc = 'googlenet_DLN' # model architecture
    extra_id = '_740'
    device_id = 7 # cuda device ID

    # ['2022-06-26', '_26', 'vit']
    # ['2022-06-26', '_27', 'effnet']
    # ['2022-06-27', '_29', 'resnet']
    # ['2022-06-27', '_30', 'googlenet']
    # ['2022-06-27', '_32', 'googlenet-vggface2']

    # ['2022-07-14', '_126', 'vit']
    # ['2022-07-14', '_127', 'resnet']
    # ['2022-07-14', '_128', 'effnet']
    # ['2022-07-14', '_129', 'googlenet']

    # ['2022-07-20', '_710', 'resnet_DLN']
    # ['2022-07-21', '_720', 'vit_DLN']
    # ['2022-07-21', '_730', 'effnet_DLN']
    # ['2022-07-21', '_740', 'googlenet_DLN']

    # ['2022-07-20', '_810', 'vit_DLN'] AffectNet_pretrain epoch_10_' + '2022-07-14' + '_126' + '.pt'
    # ['2022-07-21', '_820', 'resnet_DLN'] AffectNet_pretrain epoch_10_' + '2022-07-14' + '_127' + '.pt'
    # ['2022-07-21', '_830', 'effnet_DLN'] AffectNet_pretrain epoch_10_' + '2022-07-14' + '_128' + '.pt'
    # ['2022-07-21', '_840', 'googlenet_DLN'] AffectNet_pretrain epoch_10_' + '2022-07-14' + '_129' + '.pt'

    device = torch.device('cuda:' + str(device_id))

    print('using cuda device ID:', device_id)
    print('model architecture:', model_arc)
    print('experiment ID:', extra_id)


    print('#' * 20)
    print('testing on all validation set data...')
    EXPR_test_CSV_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/all_valid.csv'
    face_sentiment_resnet(device, num_classes, batch_size, num_workers,
                          EXPR_test_CSV_path, model_arc, extra_id)

    print('#' * 20)
    print('testing on AffectNet validation set data...')
    EXPR_test_CSV_path = '/mnt/data2/sylyoung/AffectNet/val_set/valid.csv'
    face_sentiment_resnet(device, num_classes, batch_size, num_workers,
                          EXPR_test_CSV_path, model_arc, extra_id)

    print('#' * 20)
    print('testing on test set...')
    EXPR_test_CSV_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/test_data.csv'
    face_sentiment_resnet(device, num_classes, batch_size, num_workers,
                          EXPR_test_CSV_path, model_arc, extra_id, True)


