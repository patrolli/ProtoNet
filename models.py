import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import torchvision
import torch.nn as nn
from classifiers import EuclideanClassifier
from networks import ConvNet4, ResNet12
import dataload
from utils import compute_confidence_interval, loadLogger
import argparse
import torch.optim as optim
from tqdm import tqdm


class Model:
    def __init__(self, args, image_channel, out_channel, image_size, logger):
        if args.backbone == 'ResNet12':
            self.net = ResNet12(image_channel, image_size)
        else:
            self.net = ConvNet4(image_channel, out_channel, image_size)
        self.classifier = EuclideanClassifier(args)
        if torch.cuda.is_available():
            print('GPU is used!')
            logger.info('GPU is used!')
            self.models_on_cuda()
        else:
            print('cuda is not running')
            logger.info('cuda is not running')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.q_query = args.q_query
        self.dataset = args.dataset
        self.logger = logger

    def train(self, epoch, episodes, train_folders, print_each=20):
        self.net.train()
        self.classifier.train()
        loss_avg = 0
        for i in tqdm(range(episodes)):
            task = dataload.FewShotTask(train_folders, self.n_way, self.k_shot, self.q_query, self.dataset)
            spt_loader = dataload.get_Dataloader(task, self.k_shot, 'support')
            qry_loader = dataload.get_Dataloader(task, self.q_query, 'query', shuffle=True)
            spt, spt_y = iter(spt_loader).next()
            qry, qry_y = iter(qry_loader).next()
            if torch.cuda.is_available():
                spt, spt_y, qry, qry_y = self.datas_on_cuda(spt, spt_y, qry, qry_y)

            spt_feature = self.net(spt)
            qry_feature = self.net(qry)
            dist = self.classifier(spt_feature, qry_feature)
            loss = self.loss_fn(-dist, qry_y.long())
            loss_avg += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i+1) % print_each == 0:
                print("epoch: {}\tepisodes: {}\tloss: {:.2f}".format(epoch+1, i+1, loss_avg/1.0/print_each))
                self.logger.info("epoch: {}\tepisodes: {}\tloss: {:.2f}".format(epoch+1, i+1, loss_avg/1.0/print_each))
                loss_avg = 0

    def val(self, episodes, val_folders):
        self.net.eval()
        self.classifier.eval()
        accs = []
        for i in range(episodes):
            task = dataload.FewShotTask(val_folders, self.n_way, self.k_shot, self.q_query, self.dataset)
            spt_loader = dataload.get_Dataloader(task, self.k_shot, 'support')
            qry_loader = dataload.get_Dataloader(task, self.q_query, 'query', shuffle=True)
            spt, spt_y = iter(spt_loader).next()
            qry, qry_y = iter(qry_loader).next()
            if torch.cuda.is_available():
                spt, spt_y, qry, qry_y = self.datas_on_cuda(spt, spt_y, qry, qry_y)

            spt_feature = self.net(spt)
            qry_feature = self.net(qry)
            dist = self.classifier(spt_feature, qry_feature)
            predict_label = torch.argmin(dist, dim=1)
            score = [1 if predict_label[j] == qry_y[j].long() else 0 for j in range(len(qry_y))]
            acc = np.sum(score) / 1.0 / len(score)
            accs.append(acc)
        acc, pacc = compute_confidence_interval(accs)
        print('acc is {:.4f} pacc is {:.4f}'.format(acc, pacc))
        self.logger.info('acc is {:.4f} pacc is {:.4f}'.format(acc, pacc))
        return acc, pacc

    def models_on_cuda(self):
        self.net.cuda()

    def datas_on_cuda(self, spt, spt_y, qry, qry_y):
        spt = spt.cuda()
        spt_y = spt_y.cuda()
        qry = qry.cuda()
        qry_y = qry_y.cuda()
        return spt, spt_y, qry, qry_y



