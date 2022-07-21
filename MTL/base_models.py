import pytorch_lightning as pl
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from data_utils import get_metric_func
import numpy as np
# from miners.triplet_margin_miner import TripletMarginMiner

import math


class AUClassifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, seq_input):
        bs, seq_len = seq_input.size(0), seq_input.size(1)
        weight = self.fc.weight
        bias = self.fc.bias
        seq_input = seq_input.reshape((bs * seq_len, 1, -1))  # bs*seq_len, 1, metric_dim
        weight = weight.unsqueeze(0).repeat((bs, 1, 1))  # bs,seq_len, metric_dim
        weight = weight.view((bs * seq_len, -1)).unsqueeze(-1)  # bs*seq_len, metric_dim, 1
        inner_product = torch.bmm(seq_input, weight).squeeze(-1).squeeze(-1)  # bs*seq_len
        inner_product = inner_product.view((bs, seq_len))
        return inner_product + bias


class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx, dataloader_idx):
        raise NotImplementedError

    def validation_epoch_end(self, validation_step_outputs):
        raise NotImplementedError

    def configure_optimizers(self):
        parameters_dict = [{'params': self.parameters(), 'lr': self.lr}]
        optimizer = torch.optim.SGD(parameters_dict, momentum=0.9, lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }  # cosine annealing scheduler


class MultitaskModel(Model):
    def __init__(self, *args, **kwargs):
        super(MultitaskModel, self).__init__(*args, **kwargs)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # the batch input: input_image, label
        x, y = batch
        preds = self(x)  # the batch output: preds_dictionary, metrics
        return torch.cat([preds['AU'], preds['EXPR'], preds['VA']], dim=-1), y


class InceptionV3MTModel(MultitaskModel):
    # the initialization function is shared by all models with InceptionV3 feature extractor
    def __init__(self, tasks: List[str],
                 au_names_list: List[str], emotion_names_list: List[str], va_dim: int,
                 AU_metric_dim: int,
                 n_heads: int = 8,
                 dropout=0.3,
                 lr: float = 1e-3,
                 T_max: int = 1e4, wd: float = 0.,
                 AU_cls_loss_func=None,
                 EXPR_cls_loss_func=None,
                 VA_cls_loss_func=None):
        super(InceptionV3MTModel, self).__init__()
        self.tasks = tasks
        self.backbone_CNN = inception_v3(pretrained=True)
        self.au_names_list = au_names_list
        self.emotion_names_list = emotion_names_list
        self.va_dim = va_dim
        self.AU_metric_dim = AU_metric_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.T_max = T_max
        self.wd = wd
        self.AU_cls_loss_func = AU_cls_loss_func
        self.EXPR_cls_loss_func = EXPR_cls_loss_func
        self.VA_cls_loss_func = VA_cls_loss_func
        self.N_emotions = len(self.emotion_names_list)
        self.features_dim = self.backbone_CNN.features_dim
        self.configure_architecture()  # define unique model architecture

    def training_task(self, preds_task, labels_task, task):
        cls_loss_func = getattr(self, task + '_cls_loss_func')
        cls_loss_values = cls_loss_func(preds_task, labels_task)
        return cls_loss_values

    def training_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        (x_au, y_au), (x_expr, y_expr), (x_va, y_va) = batch
        total_loss = 0
        for task in self.tasks:
            if task == 'AU':
                preds = self(x_au)
                preds_task = preds[task]
                labels_task = y_au
            elif task == 'EXPR':
                preds = self(x_expr)
                preds_task = preds[task]
                labels_task = y_expr
            elif task == 'VA':
                preds= self(x_va)
                preds_task = preds[task]
                labels_task = y_va
            else:
                raise ValueError
            loss = self.training_task(preds_task, labels_task, task)

            self.log('loss_{}: '.format(task), loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True)
            total_loss += loss
        self.log('total_loss: ', total_loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return total_loss

    def save_val_metrics_task(self, preds_task, labels_task, save_name, task):
        metric_values, _ = get_metric_func(task)(preds_task.numpy(), labels_task.numpy())
        if task != 'VA':
            self.log('{}_F1'.format(save_name), metric_values[0], on_epoch=True, logger=True)
            self.log('{}_Acc'.format(save_name), metric_values[1], on_epoch=True, logger=True)
            return metric_values[0]  # return F1 for EXPR and AU
        else:
            self.log('{}_{}'.format(save_name, 'valence'), metric_values[0], on_epoch=True, logger=True)
            self.log('{}_{}'.format(save_name, 'arousal'), metric_values[1], on_epoch=True, logger=True)
            return 0.5 * metric_values[0] + 0.5 * metric_values[1]

    def turn_mul_emotion_preds_to_dict(self, preds):
        # au, expr , va
        return {'AU': preds[..., :len(self.au_names_list)],
                'EXPR': preds[..., len(self.au_names_list): len(self.au_names_list) + len(self.emotion_names_list)],
                'VA': preds[..., -2:]}

    def validation_epoch_end(self, validation_step_outputs):
        # check the validation step outputs
        num_dataloaders = len(validation_step_outputs)
        total_metric = 0
        for dataloader_idx in range(num_dataloaders):
            val_dl_outputs = validation_step_outputs[dataloader_idx]
            idx_metric = self.validation_on_single_dataloader(dataloader_idx, val_dl_outputs)
            total_metric += idx_metric
        self.log('val_total', total_metric, on_epoch=True, logger=True)

    def validation_on_single_dataloader(self, dataloader_idx, val_dl_outputs):
        num_batches = len(val_dl_outputs)
        preds = torch.cat([x[0] for x in val_dl_outputs], dim=0).cpu()
        labels = torch.cat([x[1] for x in val_dl_outputs], dim=0).cpu()
        preds = self.turn_mul_emotion_preds_to_dict(preds)
        idx_metric = self.validation_single_task(dataloader_idx, preds, labels)
        return idx_metric

    def validation_single_task(self, dataloader_idx, preds, labels):
        if labels.size(1) == 1:
            # expr
            preds_task = F.softmax(preds['EXPR'], dim=-1).argmax(-1).int()
            labels_task = labels.int()
            metric_8_task = self.save_val_metrics_task(preds_task,
                                                       labels_task, "D{}/EXPR8".format(dataloader_idx), 'EXPR')
            return metric_8_task

        elif labels.size(1) == 12:
            # au
            preds_task = (torch.sigmoid(preds['AU']) > 0.5).int()
            labels_task = labels.int()
            metric_aus = self.save_val_metrics_task(preds_task,
                                                    labels_task, "D{}/AU".format(dataloader_idx), 'AU')
            return metric_aus
        elif labels.size(1) == 2:
            # va
            preds_task = preds['VA'].float()
            labels_task = labels
            metric_va = self.save_val_metrics_task(preds_task,
                                                   labels_task, "D{}/VA".format(dataloader_idx), 'VA')
            return metric_va


def inception_v3(pretrained=True):
    CNN = torchvision.models.inception_v3(pretrained=pretrained)

    layers_to_keep = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
                      'maxpool1', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'maxpool2',
                      'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
                      'Mixed_6d', 'Mixed_6e']
    layers_to_keep = [getattr(CNN, name) for name in layers_to_keep]
    CNN = torch.nn.Sequential(*layers_to_keep)
    setattr(CNN, 'features_dim', 768)
    setattr(CNN, 'features_width', 17)
    return CNN
