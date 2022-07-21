import torchvision.transforms as transforms
import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def averaged_f1_score(input, target):
    N, label_size = input.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(input[:, i], target[:, i])
        f1s.append(f1)
    return np.mean(f1s), f1s

def accuracy(input, target):
    assert len(input.shape) == 1
    return np.sum(input==target)/input.shape[0]

def averaged_accuracy(x, y):
    assert len(x.shape) == 2
    N, C =x.shape
    accs = []
    for i in range(C):
        acc = accuracy(x[:, i], y[:, i])
        accs.append(acc)
    return np.mean(accs), accs

EPS = 1e-8

def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def weighted_correlation(x, y, weight):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    rho = torch.sum(vx * vy * torch.pow(weight, 2)) / (torch.sqrt(torch.sum(torch.pow(vx * weight, 2))) *
                                                       torch.sqrt(torch.sum(torch.pow(vy * weight, 2))) + EPS)
    return rho

def weighted_std(x,  weight):
    vx = x - torch.mean(x)
    return torch.sqrt(torch.sum(torch.pow(vx,2)*weight)/torch.sum(weight))

def weighted_mean(x, weight):
    return torch.sum(x*weight)/torch.sum(weight)

def VA_metric(x, y):
    items = [CCC_score(x[:,0], y[:,0]), CCC_score(x[:,1], y[:,1])]
    return items, sum(items)
def EXPR_metric(x, y): 
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)

    f1 = f1_score(x, y, average= 'macro')
    acc = accuracy(x, y)
    return [f1, acc], 0.67*f1 + 0.33*acc
def AU_metric(x, y):
    f1_av, f1s = averaged_f1_score(x, y)
    x = x.reshape(-1)
    y = y.reshape(-1)
    acc_av  = accuracy(x, y)
    return [f1_av, acc_av, f1s], 0.5*f1_av + 0.5*acc_av

class CCCLoss(nn.Module):
    def __init__(self, digitize_num=20, range=[-1, 1], weight=None):
        super(CCCLoss, self).__init__() 
        self.digitize_num =  digitize_num
        self.range = range
        self.weight = weight
        if self.digitize_num >1:
            bins = np.linspace(*self.range, num= self.digitize_num)
            self.bins = torch.as_tensor(bins, dtype = torch.float32).cuda().view((1, -1))
    def forward(self, x, y): 
        # the target y is continuous value (BS, )
        # the input x is either continuous value (BS, ) or probability output(digitized)
        y = y.view(-1)
        if self.digitize_num !=1:
            x = F.softmax(x, dim=-1)
            x = (self.bins * x).sum(-1) # expectation
        x = x.view(-1)
        if self.weight is None:
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + EPS)
            x_m = torch.mean(x)
            y_m = torch.mean(y)
            x_s = torch.std(x)
            y_s = torch.std(y)
            ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + EPS)
        else:
            rho = weighted_correlation(x, y, self.weight)
            x_var = weighted_std(x, self.weight)
            y_var = weighted_std(y, self.weight)
            x_mean = weighted_mean(x, self.weight)
            y_mean = weighted_mean(y, self.weight)
            ccc = 2*rho*torch.sqrt(x_var)*torch.sqrt(y_var)/(x_var + y_var + torch.pow(x_mean - y_mean, 2) +EPS)
        return 1-ccc


def get_metric_func(task):
    if task =='VA':
        return VA_metric
    elif task=='EXPR':
        return EXPR_metric
    elif task=='AU':
        return AU_metric
        
def train_transforms(img_size):
    transform_list = [transforms.Resize([int(img_size*1.02), int(img_size*1.02)]),
                      transforms.RandomCrop([img_size, img_size]),
                      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
                    ]
    return transforms.Compose(transform_list)

def test_transforms(img_size):
    transform_list = [transforms.Resize([img_size, img_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
                    ]
    return transforms.Compose(transform_list)

def extact_face_landmarks(df):
    face_shape = []
    for i in range(1, 69):
        x,  y = df['x_{}'.format(i)], df['y_{}'.format(i)]
        x, y = int(x), int(y)
        face_shape.append([x, y])
    return np.array(face_shape)

def transform_corrdinate(x, y, in_size, out_size):
    new_x, new_y = x * (out_size[0]/in_size[0]), y * (out_size[1]/in_size[1])
    new_x, new_y = int(new_x), int(new_y)
    assert new_x< out_size[0] and new_y< out_size[1]
    return new_x, new_y

def blurry_image(input_img):
    return gaussian_filter(input_img, sigma=3)
