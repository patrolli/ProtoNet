import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import torchvision
import torch.nn as nn
from classifiers import EuclideanClassifier
from networks import ConvNet4
import dataload
import argparse
import torch.optim as optim
from models import Model
from torch.optim.lr_scheduler import StepLR
from utils import loadLogger

image_folders = {'Omniglot': {
                    'train': '../MachineLearning/data/Omniglot/images_background',
                    'test': '../MachineLearning/data/Omniglot/images_evaluation'},
                 'miniimagenet': {
                     'train': '/opt/data/private/FSL_Datasets/miniimagenet/train',
                     'val': '/opt/data/private/FSL_Datasets/miniimagenet/val',
                     'test': '/opt/data/private/FSL_Datasets/miniimagenet/test'
                 }}

log_dir = './logs'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
logger = loadLogger(log_dir, mode='info', not_save=False)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_way", type=int, default=5)
    parser.add_argument('-k', '--k_shot', type=int, default=5)
    parser.add_argument('-q', '--q_query', type=int, default=5)
    parser.add_argument('-e', '--episodes', type=int, default=200)
    parser.add_argument('--epoch', type=int, default=50)    # all episodes = epoch*episodes
    parser.add_argument('-d', '--dataset', type=str, default='miniimagenet')
    parser.add_argument('-r', '--resume', action="store_true")
    parser.add_argument('-b', '--backbone', type=str, default='ResNet12')
    args = parser.parse_args()

    image_channel = 1 if args.dataset == 'Omniglot' else 3
    image_size = 28 if args.dataset == 'Omniglot' else 84

    out_channel = 64

    save_name = args.dataset+'_' + str(args.backbone) + '_'+str(args.n_way)+'_way'+'_'+str(args.k_shot)+\
                '_shot'+'_'+str(args.q_query)+'_query'
    save_path = './checkpoint/'+ save_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_folders = dataload.get_folders(image_folders[args.dataset]['train'])
    if args.dataset == 'miniimagenet':
        val_folders = dataload.get_folders(image_folders[args.dataset]['val'])
    else:
        val_folders = dataload.get_folders(image_folders[args.dataet]['test'])
    test_folders = dataload.get_folders(image_folders[args.dataset]['test'])


    best_acc = 0
    epoch = 0
    model = Model(args, image_channel, out_channel, image_size, logger)
    if args.resume:
        print('continuing training...')
        checkpoint = torch.load(os.path.join(save_path, save_name+'.pth'))
        epoch = checkpoint['epoch']
        model.net = checkpoint['model']
        model.optimizer = checkpoint['optimizer']
        best_acc = checkpoint['best_acc']

    scheduler = StepLR(model.optimizer, step_size=int(2000/args.episodes), gamma=0.5)  # each 2000 episodes to update

    # begin training
    for epoch in range(epoch, args.epoch):
        model.train(epoch, args.episodes, train_folders, print_each=20)
        # acc, _ = model.val(1000, val_folders)
        # if acc > best_acc:
        #    best_acc = acc
        #    checkpoint = {
        #        'best_acc': best_acc,
        #        'episode': args.episodes,
        #        'epoch': epoch+1,
        #        'model': model.net.state_dict(),
        #        'optimizer': model.optimizer.state_dict()
        #    }
        #    torch.save(checkpoint, os.path.join(save_path, save_name+'.pth'))
        scheduler.step()

        checkpoint = {
            'best_acc': 0,
            'episode': args.episodes,
            'epoch': args.epoch,
            'model': model.net.state_dict(),
            'optimizer': model.optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(save_path, save_name + '.pth'))


if __name__ == '__main__':
    main()