"""
@author: Siyang Li
@date: Jun.13 2022
@email: lsyyoungll@gmail.com

ABAW4 Challenge
Learning from Synthetic data
DLN method: Fine-tuning SOTA feature extractors with DLN

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

#from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import Dataset, DataLoader
from skimage import io
from sklearn.metrics import accuracy_score, f1_score
from datetime import date
from tqdm import tqdm
#from torchsummary import summary
from torchvision.models import resnet50, inception_v3, googlenet, vit_b_16, efficientnet_b0, vgg16, vit_b_32, convnext_tiny


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


class FC_classifier(nn.Module):

    def __init__(self, fc_num=0, out_chann=0):
        super(FC_classifier, self).__init__()

        # FC Layer
        self.fc = nn.Linear(fc_num, out_chann)

    def forward(self, x):
        x = self.fc(x)
        return x


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
        label = int(self.csv_file.iloc[idx, 1])
        label = torch.Tensor([label]).to(torch.long)

        return image, label


def face_sentiment(device, num_classes=None, batch_size=None, savelog=True, lr=0.001, num_iter=10, num_workers=1,
                          csv_path_train=None, csv_path_valid=None, csv_path_test=None, extra_id='', model_arc=None):


    if model_arc == 'vit':
        model_identity = vit_b_16(pretrained=True)
        del model_identity.heads
        for para in model_identity.parameters():
            para.requires_grad = False
        model_face = vit_b_16(pretrained=True)
        del model_face.heads
        for para in model_face.parameters():
            para.requires_grad = True
        clf = FC_classifier(768, 6)
        model_identity.load_state_dict(
            torch.load('./runs/baseline/model_feature_epoch_10_' + '2022-07-14' + '_126' + '.pt'))  # 07-04 38
        model_face.load_state_dict(
            torch.load('./runs/baseline/model_feature_epoch_10_' + '2022-07-14' + '_126' + '.pt'))  # 07-04 38
    elif model_arc == 'effnet':
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
        current_model_dict = model_identity.state_dict()
        loaded_state_dict = torch.load('./runs/baseline/model_epoch_10_' + '2022-07-14' + '_128' + '.pt')
        new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
                          zip(current_model_dict.keys(), loaded_state_dict.values())}
        model_identity.load_state_dict(new_state_dict, strict=False)
        model_face.load_state_dict(new_state_dict, strict=False)
    elif model_arc == 'resnet':
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
        current_model_dict = model_identity.state_dict()
        loaded_state_dict = torch.load('./runs/baseline/model_epoch_10_' + '2022-07-14' + '_127' + '.pt')
        new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
                          zip(current_model_dict.keys(), loaded_state_dict.values())}
        model_identity.load_state_dict(new_state_dict, strict=False)
        model_face.load_state_dict(new_state_dict, strict=False)
    elif model_arc == 'googlenet':
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
        current_model_dict = model_identity.state_dict()
        loaded_state_dict = torch.load('./runs/baseline/model_epoch_10_' + '2022-07-14' + '_129' + '.pt')
        new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
                          zip(current_model_dict.keys(), loaded_state_dict.values())}
        model_identity.load_state_dict(new_state_dict, strict=False)
        model_face.load_state_dict(new_state_dict, strict=False)

    model_identity.to(device)
    model_face.to(device)
    clf.to(device)

    img_size = 224

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize([img_size, img_size]),
         transforms.Normalize(mean=(0.4734, 0.3544, 0.3267), std=(0.1891, 0.1597, 0.1504))])

    train_dataset = CustomImagesDataset(csv_path_train, transform=transform, device=device)
    valid_dataset = CustomImagesDataset(csv_path_valid, transform=transform, device=device)
    test_dataset = CustomImagesDataset(csv_path_test, transform=transform, device=device)

    train_iterator = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True, drop_last=True)
    valid_iterator = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_iterator = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    def train(model_identity, model_face, clf, iterator, optimizer, criterion, savelog=False, epoch=None, exp_time=None):

        print('training...')

        epoch_loss = 0
        epoch_acc = 0

        model_identity.eval()
        model_face.eval()
        clf.train()

        eps = None

        for batch in tqdm(iterator):
            optimizer.zero_grad()

            data, labels = batch
            data, labels = data.to(device), labels.to(device)

            if eps is None:
                eps = torch.ones(model_identity(data).shape).to(device) * 1e-06

            out = model_face(data) - model_identity(data) + eps

            while len(out.shape) > 2:
                out = torch.squeeze(out, 2)

            predictions = clf(out)
            _, predicted = torch.max(predictions.cpu().data, 1)

            loss = criterion(predictions, labels.reshape(-1, ))
            acc = accuracy_score(predicted, labels.detach().cpu())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

        avg_loss, avg_acc = epoch_loss / len(iterator), epoch_acc / len(iterator)

        if savelog:
            with open('./runs/baseline/log_train_' + exp_time + extra_id, 'a') as logfile:
                logfile.write('Epoch ' + str(epoch) + ': Train Loss ' + str(round(avg_loss, 5)) + ', Train Acc ' + str(
                    round(avg_acc, 5)))
                logfile.write('\n')

        return avg_loss, avg_acc

    def evaluate(model_identity, model_face, clf, iterator, criterion, savelog=False, phase=None, epoch=None, exp_time=None):

        print('evaluating...')

        epoch_loss = 0

        model_identity.eval()
        model_face.eval()
        clf.eval()

        predicted_all = []
        labels_all = []

        with torch.no_grad():
            for batch in iterator:
                data, labels = batch
                data, labels = data.to(device), labels.to(device)

                out = model_face(data) - model_identity(data)

                while len(out.shape) > 2:
                    out = torch.squeeze(out, 2)

                predictions = clf(out)
                _, predicted = torch.max(predictions.cpu().data, 1)

                loss = criterion(predictions, labels.reshape(-1, ))

                predicted_all.append(predicted)
                labels_all.append(labels.detach().cpu())

                epoch_loss += loss.item()

        predicted_all = np.concatenate(predicted_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        predicted_all = predicted_all.reshape(-1, 1)
        labels_all = labels_all.reshape(-1, 1)
        f1_arr = f1_score(labels_all, predicted_all, average=None)
        epoch_acc = accuracy_score(labels_all, predicted_all)

        avg_loss = epoch_loss / len(iterator)

        if savelog and phase == 'Valid':
            with open('./runs/baseline/log_train_' + exp_time + extra_id, 'a') as logfile:
                logfile.write('Epoch ' + str(epoch) + ': Valid Loss ' + str(round(avg_loss, 5)) + ', Valid Acc ' +
                              str(round(epoch_acc, 5)) + ', Valid F1 ' + str(round(np.average(f1_arr), 5)))
                logfile.write('\n')

        if savelog and phase == 'Test':
            with open('./runs/baseline/log_test_' + exp_time + extra_id, 'a') as logfile:
                logfile.write('Test Loss ' + str(round(avg_loss, 5)) + ', Test Acc ' + str(round(epoch_acc, 5))
                              + ', Test F1 ' + str(round(np.average(f1_arr), 5)))
                logfile.write('\n')

        return avg_loss, epoch_acc, f1_arr

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    optimizer = optim.Adam(list(model_face.parameters()) + list(clf.parameters()), lr=lr)

    class_weight = torch.tensor([1,3,5,1,1,1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.2)

    criterion_eval = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')

    experiment_time = str(date.today())
    print("Current date:", experiment_time)

    # training phase
    # early stopping (model save) with valid set
    for epoch in range(num_iter):
        print('epoch :' + str(epoch + 1) + ' of ' + str(num_iter))

        start_time = time.time()

        train_loss, train_acc = train(model_identity, model_face, clf, train_iterator, optimizer, criterion,
                                      savelog=savelog, epoch=(epoch + 1), exp_time=experiment_time)
        valid_loss, valid_acc, f1_arr = evaluate(model_identity, model_face, clf, valid_iterator, criterion_eval,
                                                 savelog=savelog, phase='Valid', epoch=(epoch + 1),
                                                 exp_time=experiment_time)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('Current epoch is best epoch for valid set, saving models...')
            torch.save(model_face.state_dict(), './runs/baseline/model_face_best_' + experiment_time + extra_id + '.pt')
            torch.save(clf.state_dict(), './runs/baseline/model_clf_best_' + experiment_time + extra_id + '.pt')

        if (epoch + 1) % 5 == 0:
            torch.save(model_face.state_dict(),
                       './runs/baseline/model_face_epoch_' + str(epoch + 1) + '_' + experiment_time + extra_id + '.pt')
            torch.save(clf.state_dict(),
                       './runs/baseline/model_clf_epoch_' + str(epoch + 1) + '_' + experiment_time + extra_id + '.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss/Accuracy:  {train_loss:.5f} {train_acc:.5f}')
        print(f'\t Val. Loss/Accuracy/F1: {valid_loss:.5f} {valid_acc:.5f} {np.average(f1_arr):.5f}')
        print(f'\t Val. F1: ', f1_arr)

    # test phase below
    # modify experiment_time if needed
    model_face.load_state_dict(
        torch.load('./runs/baseline/model_face_best_' + experiment_time + extra_id + '.pt'))
    clf.load_state_dict(
        torch.load('./runs/baseline/model_clf_best_' + experiment_time + extra_id + '.pt'))

    test_loss, test_acc, f1_arr = evaluate(model_identity, model_face, clf, test_iterator, criterion_eval, savelog=savelog, phase='Test',
                                           exp_time=experiment_time)

    print(f'Test Loss/Accuracy/F1: {test_loss:.5f} {test_acc:.5f} {np.average(f1_arr):.5f}')
    print(f'Test F1: ', f1_arr)


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    cuda = True

    device = torch.device('cpu')
    if cuda:
        device = torch.device('cuda:5')
        print('using cuda...')

    # whether to save running outputs (loss/acc)
    savelog = True

    lr = 0.000001
    num_iter = 10
    num_classes = 6
    batch_size = 64
    num_workers = 8
    model_arc = 'googlenet'
    extra_id = '_840'

    EXPR_train_data_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/training_set_synthetic_images'
    EXPR_test_data_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/validation_set_real_images'

    EXPR_train_CSV_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/randaug_combined.csv'
    EXPR_test_CSV_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/all_valid.csv'
    EXPR_valid_CSV_path = '/mnt/data2/sylyoung/AffectNet/val_set/valid.csv'


    face_sentiment(device, num_classes, batch_size, savelog, lr, num_iter, num_workers,
                          EXPR_train_CSV_path, EXPR_valid_CSV_path, EXPR_test_CSV_path, extra_id, model_arc)
