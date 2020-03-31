import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import torchvision.transforms as transforms


def get_folders(data_path):
    folders = []
    for _dir in os.listdir(data_path):
        dirpath = os.path.join(data_path, _dir)
        if 'mini' in data_path:
            folders.append(dirpath)
        else:
            for character in os.listdir(dirpath):
                folders.append(os.path.join(dirpath, character))
    return folders


class FewShotTask(object):
    def __init__(self, cls_folders, n_way, k_shot, q_num, dataset):
        self.cls_folders = cls_folders
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_num = q_num
        self.dataset = dataset

        class_folders = random.sample(self.cls_folders, self.n_way)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        sample = {}
        self.support_roots = []
        self.query_roots = []
        for c in class_folders:
            files = [os.path.join(c, x) for x in os.listdir(c)]
            sample[c] = random.sample(files, len(files))

            self.support_roots += sample[c][:k_shot]
            self.query_roots += sample[c][k_shot:k_shot + q_num]

            self.support_labels = [labels[self.get_class(x)] for x in self.support_roots]
            self.query_labels = [labels[self.get_class(x)] for x in self.query_roots]

    def get_class(self, root):
        return os.path.join('/', *root.split('/')[:-1])


class FewShotDataset(Dataset):
    def __init__(self, task, split='support', transform=None):
        self.transform = transform
        self.task = task
        self.image_roots = task.support_roots if split == 'support' else task.query_roots
        self.labels = task.support_labels if split == 'support' else task.query_labels
        self.split = split

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, item):
        raise NotImplementedError("Not Implemented")


class SubDataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(SubDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, item):
        image_root = self.image_roots[item]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[item]
        return image, label


class EpisodeSampler(Sampler):
    def __init__(self, n_way, num_per_class, shuffle=True):
        self.n_way = n_way
        self.num_per_class = num_per_class
        self.shuffle = shuffle

    def __iter__(self):
        indexes = list(range(self.n_way*self.num_per_class))
        if self.shuffle is True:
            random.shuffle(indexes)
        return iter(indexes)


def get_Dataloader(task, num_per_class=1, split='support', shuffle=False, transform=None):
    if transform is None:
        if task.dataset == 'Omniglot':
            default_transform = transforms.Compose([transforms.Resize((28, 28)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.5,), std=(0.5,))])
        elif task.dataset == 'miniimagenet':
            default_transform = transforms.Compose([
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        dataset = SubDataset(task, split, default_transform)
    else:
        dataset = SubDataset(task, split, transform)

    if split == 'support':
        sampler = EpisodeSampler(task.n_way, task.k_shot, shuffle=shuffle)
    else:
        sampler = EpisodeSampler(task.n_way, task.q_num, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=task.n_way*num_per_class, sampler=sampler)
    return loader