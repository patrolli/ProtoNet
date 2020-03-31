import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


# load data
file_dir = '../../FSL_Datasets/miniimagenet'

data_transform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(file_dir, x), data_transform[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=8)
               for x in ['train', 'val']}
dataset_size = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_name = image_datasets['train'].classes
print(dataset_size['train'], dataset_size['val'])
print('train classes num:{} val classes num: {}'.format(len(image_datasets['train'].classes), len(image_datasets['val'].classes)))


# TODO: ResNet-12 with attention block
# This refers with https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes//ratio, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//ratio, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = self.sigmoid(avg_out)
        return out


class Block(nn.Module):
    def __init__(self, in_planes, planes, downsample=None):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(2)
        self.ca = ChannelAttention(planes)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ca(out) * out  # pay attention on feature map
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        out = self.maxpool(out)
        return out


class ResNet12CA(nn.Module):
    def __init__(self, channels, num_classes=64):
        super(ResNet12CA, self).__init__()
        self.inplanes = 3
        self.outdim = channels[3]
        self.layer1 =self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.outdim, num_classes)

    def _make_layer(self, planes):
        downsample =nn.Sequential(
            nn.Conv2d(self.inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        block = Block(self.inplanes, planes, downsample=downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)
        o = self.avg_pool(o)
        o = o.view(o.size(0), -1)
        o = self.fc(o)
        return o


resnet12 = ResNet12CA([64, 128, 256, 512])
resnet12.cuda()
optimizer = optim.Adam(resnet12.parameters(), lr=0.001)
lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
epochs = 100
loss_fn = nn.CrossEntropyLoss()
# TODO: Train loop

for epoch in tqdm(range(epochs)):
    for i, batch in enumerate(dataloaders['train']):
        data, label = batch
        data = data.cuda()
        label = label.cuda()
        out = resnet12(data)
        loss = loss_fn(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 50 == 0:
            print("epoch:{} batch:{} loss:{:.3f}".format(epoch+1, i+1, loss.item()))
    lr_scheduler.step()

save_file = './pretrain_attention'
save_name = 'epoch_'+str(epochs) + '_batch_size_' + str(64) + '.pth'
if not os.path.exists(save_file):
    os.mkdir(save_file)
torch.save(resnet12.state_dict(), os.path.join(save_file, save_name))

