"""
@author: Siyang Li
@date: Jun.13 2022
@email: lsyyoungll@gmail.com

ABAW4 Challenge
Learning from Synthetic data
Baseline method: Fine-tuning SOTA feature extractors

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

from torch.utils.data import Dataset, DataLoader
from skimage import io
from sklearn.metrics import accuracy_score, f1_score
from datetime import date
from tqdm import tqdm
from torchvision.models import resnet50, inception_v3, googlenet, vit_b_16, efficientnet_b0, vgg16, vit_b_32, convnext_tiny
from torch.autograd import Variable


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


def mixup_data(x, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    y_a = y_a.reshape(-1, )
    y_b = y_b.reshape(-1, )
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixaugment_criterion(criterion, pred, y_a, y_b, lam, ls_alpha):
    y_a = y_a.reshape(-1, )
    y_b = y_b.reshape(-1, )
    pred_real = pred[:len(pred) // 2]
    pred_mix = pred[len(pred) // 2:]
    c = nn.CrossEntropyLoss(label_smoothing=ls_alpha)
    return 0.5 * (lam * criterion(pred_mix, y_a) + (1 - lam) * criterion(pred_mix, y_b) + c(pred_real, y_a))


def belief_criterion(predictions, labels, ls_alpha, device):

    _, predicted = torch.max(predictions, 1)

    c = nn.CrossEntropyLoss(label_smoothing=ls_alpha, reduction='sum')

    x1 = torch.where(predicted == labels)

    x2 = torch.where(predicted != labels)

    x1loss = c(predictions[x1[0]], labels[x1[0]])

    x2loss = torch.tensor(0.3).to(device) * c(predictions[x2[0]], predicted[x2[0]]) + torch.tensor(0.7).to(device) * c(predictions[x2[0]], labels[x2[0]])

    belief_loss = (x1loss + x2loss) / len(labels)

    return belief_loss


def face_sentiment_resnet(device, num_classes=None, batch_size=None, savelog=True, lr=0.001, num_iter=10, num_workers=0,
                          csv_path_train=None, csv_path_valid=None, csv_path_test=None, extra_id=None, model_arc=None):

    train_all = False
    mixup = False
    mixaugment = False
    belief = False
    load = True
    randaug = False
    label_smoothing_amount = 0.2
    ls_alpha = 0


    if model_arc == 'vit':
        feature = vit_b_16(pretrained=True)
        del feature.heads
        if not train_all:
            for para in feature.parameters():
                para.requires_grad = False
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
        if not train_all:
            for para in model.parameters():
                para.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(1280, num_classes),
        )
        for para in model.classifier.parameters():
            para.requires_grad = True
        model.to(device)
    elif model_arc == 'resnet':
        model = resnet50(pretrained=True)
        if not train_all:
            for para in model.parameters():
                para.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        for para in model.fc.parameters():
            para.requires_grad = True
        model.to(device)
    elif model_arc == 'googlenet':
        model = googlenet(pretrained=True)
        if not train_all:
            for para in model.parameters():
                para.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        for para in model.fc.parameters():
            para.requires_grad = True
        model.to(device)

    if load:
        if type(model) is list:
            model[0].load_state_dict(
                torch.load('./runs/baseline/model_feature_epoch_10_' + '2022-07-14' + '_126' + '.pt')) # 07-04 38
            model[1].load_state_dict(
                torch.load('./runs/baseline/model_clf_epoch_10_' + '2022-07-14' + '_126' + '.pt'))
        else:
            model.load_state_dict(
                torch.load('./runs/baseline/model_epoch_10_' + '2022-07-14' + '_126' + '.pt'))

    img_size = 224
    if model_arc == 'inceptionv3':
        img_size = 299

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize([img_size, img_size]),
         #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]) # ImageNet
         transforms.Normalize(mean=(0.4734, 0.3544, 0.3267), std=(0.1891, 0.1597, 0.1504))])  # training set

    if randaug:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize([img_size, img_size]),
             transforms.ConvertImageDtype(torch.uint8),
             transforms.RandAugment(num_ops=2, magnitude=9),
             transforms.ConvertImageDtype(torch.float32),
             transforms.Normalize(mean=(0.4734, 0.3544, 0.3267), std=(0.1891, 0.1597, 0.1504))])  # training set
    else:
        train_transform = transform

    train_dataset = CustomImagesDataset(csv_path_train, transform=transform, device=device)
    valid_dataset = CustomImagesDataset(csv_path_valid, transform=transform, device=device)
    test_dataset = CustomImagesDataset(csv_path_test, transform=transform, device=device)

    train_iterator = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                shuffle=True)
    valid_iterator = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_iterator = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


    def train(model, iterator, optimizer, criterion, savelog=False, epoch=None, exp_time=None):

        print('training...')

        epoch_loss = 0
        epoch_acc = 0

        correct = 0
        total = 0

        if type(model) is list:
            model[0].train()
            model[1].train()
        else:
            model.train()

        for batch in tqdm(iterator):
            optimizer.zero_grad()

            data, labels = batch
            data, labels = data.to(device), labels.to(device)

            if mixup:
                mixed_x, y_a, y_b, lam = mixup_data(data, labels, device=device, alpha=1.0)
                data, y_a, y_b = map(Variable, (mixed_x, y_a, y_b))
                if type(model) is list:
                    predictions = model[1](model[0](data))
                else:
                    predictions = model(data)
            elif mixaugment:
                mixed_x, y_a, y_b, lam = mixup_data(data, labels, device=device, alpha=1.0)
                data_mixed, y_a, y_b = map(Variable, (mixed_x, y_a, y_b))
                data_catted = torch.cat((data, data_mixed), dim=0)
                if type(model) is list:
                    predictions = model[1](model[0](data_catted))
                else:
                    predictions = model(data_catted)
            else:
                if type(model) is list:
                    predictions = model[1](model[0](data))
                else:
                    predictions = model(data)


            _, predicted = torch.max(predictions.cpu().data, 1)

            if mixup:
                loss = mixup_criterion(criterion, predictions, y_a, y_b, lam)
                correct += (lam * predicted.eq(y_a.cpu().data).sum().float()
                            + (1 - lam) * predicted.eq(y_b.cpu().data).sum().float())
                total += labels.size(0)
            elif mixaugment:
                loss = mixaugment_criterion(criterion, predictions, y_a, y_b, lam, ls_alpha=ls_alpha)
                correct += (lam * predicted.eq(y_a.cpu().data).sum().float()
                            + (1 - lam) * predicted.eq(y_b.cpu().data).sum().float())
                total += labels.size(0)
            elif belief:
                loss = belief_criterion(predictions, labels.reshape(-1, ), ls_alpha, device)
                acc = accuracy_score(predicted, labels.detach().cpu())

            else:
                loss = criterion(predictions, labels.reshape(-1, ))
                acc = accuracy_score(predicted, labels.detach().cpu())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if not mixup and not mixaugment:
                epoch_acc += acc

        if mixup or mixaugment:
            avg_loss, avg_acc = epoch_loss / len(iterator), float(correct / total)
        else:
            avg_loss, avg_acc = epoch_loss / len(iterator), epoch_acc / len(iterator)

        if savelog:
            if not os.path.isdir('./runs'):
                os.mkdir('./runs')
            if not os.path.isdir('./runs/baseline'):
                os.mkdir('./runs/baseline')
            with open('./runs/baseline/log_train_' + exp_time + extra_id, 'a') as logfile:
                logfile.write('Epoch ' + str(epoch) + ': Train Loss ' + str(round(avg_loss, 5)) + ', Train Acc ' + str(
                    round(avg_acc, 5)))
                logfile.write('\n')

        print('#' * 50)

        return avg_loss, avg_acc

    def evaluate(model, iterator, criterion, savelog=False, phase=None, epoch=None, exp_time=None):

        print('evaluating...')

        epoch_loss = 0

        if type(model) is list:
            model[0].eval()
            model[1].eval()
        else:
            model.eval()

        predicted_all = []
        labels_all = []

        with torch.no_grad():
            for batch in iterator:
                data, labels = batch
                data, labels = data.to(device), labels.to(device)

                if type(model) is list:
                    predictions = model[1](model[0](data))
                else:
                    predictions = model(data)
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
            if not os.path.isdir('./runs'):
                os.mkdir('./runs')
            if not os.path.isdir('./runs/baseline'):
                os.mkdir('./runs/baseline')
            with open('./runs/baseline/log_train_' + exp_time + extra_id, 'a') as logfile:
                logfile.write('Epoch ' + str(epoch) + ': Valid Loss ' + str(round(avg_loss, 5)) + ', Valid Acc ' +
                              str(round(epoch_acc, 5)) + ', Valid F1 ' + str(round(np.average(f1_arr), 5)))
                logfile.write('\n')

        if savelog and phase == 'Test':
            if not os.path.isdir('./runs'):
                os.mkdir('./runs')
            if not os.path.isdir('./runs/baseline'):
                os.mkdir('./runs/baseline')
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

    if type(model) is list:
        optimizer = optim.Adam(list(model[0].parameters()) + list(model[1].parameters()), lr=lr)

    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing_amount)
    criterion_eval = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')

    experiment_time = str(date.today())
    print("Current date:", experiment_time)

    # training phase
    # early stopping (model save) with valid set
    for epoch in range(num_iter):
        print('epoch :' + str(epoch + 1) + ' of ' + str(num_iter))

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion,
                                      savelog=savelog, epoch=(epoch + 1), exp_time=experiment_time)
        valid_loss, valid_acc, f1_arr = evaluate(model, valid_iterator, criterion_eval,
                                                 savelog=savelog, phase='Valid', epoch=(epoch + 1),
                                                 exp_time=experiment_time)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if not os.path.isdir('./runs'):
            os.mkdir('./runs')
        if not os.path.isdir('./runs/baseline'):
            os.mkdir('./runs/baseline')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('Current epoch is best epoch for valid set, saving models...')
            if type(model) is list:
                torch.save(model[0].state_dict(),
                           './runs/baseline/model_feature_best_' + experiment_time + extra_id + '.pt')
                torch.save(model[1].state_dict(),
                           './runs/baseline/model_clf_best_' + experiment_time + extra_id + '.pt')
            else:
                torch.save(model.state_dict(), './runs/baseline/model_best_' + experiment_time + extra_id + '.pt')

        if (epoch + 1) % 5 == 0:
            if type(model) is list:
                torch.save(model[0].state_dict(), './runs/baseline/model_feature_epoch_' + str(
                    epoch + 1) + '_' + experiment_time + extra_id + '.pt')
                torch.save(model[1].state_dict(), './runs/baseline/model_clf_epoch_' + str(
                    epoch + 1) + '_' + experiment_time + extra_id + '.pt')
            else:
                torch.save(model.state_dict(), './runs/baseline/model_epoch_' + str(
                    epoch + 1) + '_' + experiment_time + extra_id + '.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss/Accuracy:  {train_loss:.5f} {train_acc:.5f}')
        print(f'\t Val. Loss/Accuracy/F1: {valid_loss:.5f} {valid_acc:.5f} {np.average(f1_arr):.5f}')
        print(f'\t Val. F1: ', f1_arr)

    # test phase
    if type(model) is list:
        model[0].load_state_dict(
            torch.load('./runs/baseline/model_feature_best_' + experiment_time + extra_id + '.pt'))
        model[1].load_state_dict(
            torch.load('./runs/baseline/model_clf_best_' + experiment_time + extra_id + '.pt'))
    else:
        model.load_state_dict(
            torch.load('./runs/baseline/model_best_' + experiment_time + extra_id + '.pt'))

    test_loss, test_acc, f1_arr = evaluate(model, test_iterator, criterion, savelog=savelog, phase='Test',
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

    # create csv files for data
    create_csv_files = True

    # whether to save running outputs (loss/acc)
    savelog = True

    lr = 0.00001
    num_iter = 20
    num_classes = 6
    batch_size = 64
    num_workers = 8

    model_arc = 'vit'
    extra_id = '_1001'  # experiment ID
    device_id = 5  # cuda device ID

    device = torch.device('cuda:' + str(device_id))

    EXPR_train_data_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/training_set_synthetic_images'
    EXPR_valid_data_path = '/mnt/data2/sylyoung/AffectNet/val_set'
    EXPR_test_data_path = '/mnt/data2/sylyoung/ABAW4/AffectNet/val_set'

    EXPR_train_CSV_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/randaug_combined.csv'
    EXPR_valid_CSV_path = '/mnt/data2/sylyoung/AffectNet/val_set/valid.csv'
    EXPR_test_CSV_path = '/mnt/data2/sylyoung/ABAW4/ABAW4-Synthetic/all_valid.csv'

    if create_csv_files:
        # train into train
        create_EXPR_csv(data_dir=EXPR_train_data_path,
                        shuffle=False,
                        train_csv_path=EXPR_train_CSV_path)
        create_EXPR_csv(data_dir=EXPR_valid_data_path,
                        shuffle=False,
                        train_csv_path=EXPR_valid_CSV_path)
        create_EXPR_csv(data_dir=EXPR_test_data_path,
                        shuffle=False,
                        train_csv_path=EXPR_test_CSV_path)

    print('experiment ID:', extra_id)
    face_sentiment_resnet(device, num_classes, batch_size, savelog, lr, num_iter, num_workers,
                          EXPR_train_CSV_path, EXPR_valid_CSV_path, EXPR_test_CSV_path, extra_id, model_arc)
