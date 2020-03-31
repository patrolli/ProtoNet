import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import torchvision
import torch.nn as nn


class EuclideanClassifier(nn.Module):
    '''
    spt_feature: (n_way*k_shot, feature_num), the order of dim0 is [class1, class1, ... class n, ... class n]
    qry_feature: (n_way*q_query, feature_num), the order is shuffled
    '''
    def __init__(self, args):
        super(EuclideanClassifier, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.q_num = args.q_query

    def forward(self, spt_feature, qry_feature):
        spt_feature = spt_feature.squeeze_().reshape(self.n_way, self.k_shot, -1)
        class_mean = torch.mean(spt_feature, dim=1)
        qry_feature_rpt = qry_feature.unsqueeze(1).repeat(1, self.n_way, 1).view(-1, qry_feature.size()[-1 ])
        class_mean_rpt = class_mean.repeat(self.n_way*self.q_num, 1)
        dist = torch.sqrt(torch.sum((qry_feature_rpt-class_mean_rpt)**2, dim=1)).view(-1, self.n_way)
        return dist
