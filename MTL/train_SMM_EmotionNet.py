import os
import pandas as pd
import numpy as np
from SMM_EmotionNet import Multitask_EmotionNet
from dataset import get_MTL_datamodule, get_test_dataset
from data_utils import CCCLoss
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn.functional as F
from predefined_args import get_predefined_vars
import csv

PRESET_VARS = get_predefined_vars()
torch.autograd.set_detect_anomaly(True)
torch.set_default_tensor_type(torch.FloatTensor)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--ckp-save-dir', type=str, default='SMM_EmotionNet')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume_ckp', type=str, default=None)
    args = parser.parse_args()
    au_names_list = PRESET_VARS['AU']
    emotion_names_list = PRESET_VARS['EXPR']
    pos_weight = torch.tensor(PRESET_VARS['AU_weight'])
    pos_weight = pos_weight.float().cuda()

    def classification_loss_func(y_hat, y):
        loss1 = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=pos_weight)
        return loss1

    def regression_loss_func(y_hat, y):
        loss1 = CCCLoss(digitize_num=1)(y_hat[:, 0], y[:, 0]) + CCCLoss(digitize_num=1)(y_hat[:, 1], y[:, 1])
        return loss1
    class_weights = torch.tensor(PRESET_VARS['EXPR_weight'])
    class_weights = class_weights.float().cuda()

    def cross_entropy_loss(y_hat, y):
        Num_classes = y_hat.size(-1)
        return F.cross_entropy(y_hat, y.squeeze().long(), weight=class_weights[:Num_classes])

    model = Multitask_EmotionNet(['AU', 'EXPR', 'VA'], au_names_list, emotion_names_list, va_dim=1,
                                 AU_metric_dim=16, lr=args.lr,
                                 AU_cls_loss_func=classification_loss_func,
                                 EXPR_cls_loss_func=cross_entropy_loss,
                                 VA_cls_loss_func=regression_loss_func,
                                 wd=0)
    img_size = 299
    dm = get_MTL_datamodule(img_size=img_size, batch_size=16, num_workers=4)
    ckp_dir = args.ckp_save_dir
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ckp_callback1 = ModelCheckpoint(monitor='val_total', mode='max', dirpath=ckp_dir,
                                    filename='model-{epoch:02d}-{val_total:.2f}',
                                    save_top_k=1, save_last=True)
    early_stop_callback = EarlyStopping(monitor="val_total", min_delta=0.00, patience=3, verbose=False, mode="max")
    tb_logger = pl_loggers.TensorBoardLogger(ckp_dir)

    trainer = Trainer(gpus=1, benchmark=True, default_root_dir=ckp_dir, logger=tb_logger, log_every_n_steps=100,
                      max_epochs=5, callbacks=[lr_monitor, ckp_callback1, early_stop_callback],
                      resume_from_checkpoint=args.resume_ckp,
                      auto_select_gpus=True)
    trainer.fit(model, datamodule=dm)

    dir_path = 'ABAW-MTL-test'
    test_list = 'MTL_Challenge_test_set_release.txt'
    test_dl = get_test_dataset(dir_path, test_list)
    trainer.test(model, dataloaders=test_dl, ckpt_path=ckp_callback1.best_model_path)
    test_df = pd.read_csv(os.path.join(dir_path, test_list))
    test_file_list = np.expand_dims(test_df.iloc[:, 0].values, 1)
    test_pred = np.load('test_pred.npz')['pred']
    final_predict = np.concatenate([test_file_list, test_pred], axis=1)
    df = pd.DataFrame(final_predict)
    df.iloc[:, 3:] = df.iloc[:, 3:].astype(int)
    with open('MTL_predictions.txt', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['image', 'valence', 'arousal', 'expression', 'aus'])
        for idx, row in df.iterrows():
            csv_writer.writerow(row.tolist())
        csv_file.close()

