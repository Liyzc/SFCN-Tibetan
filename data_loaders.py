import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import CIFAR10

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx, ...]
        image = image/255
        label = self.labels[idx]
        

        if self.transform:
            image = self.transform(image)

        return image, label
    
    @property
    def targets(self):
        return self.labels

def cifar10():
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = CIFAR10(root='E:\datasets',
                                train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root='E:\datasets',
                              train=False, download=True, transform=transform_test)
        norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        return train_dataset, val_dataset, norm

def tmnist():
        transform_train = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()])
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        '''
        raw_data = np.load('/mnt/DataDrive/liyuze/work/tibet/TibetanMNIST/Datasets/TibetanMNIST.npz')
        image_data, label_data = raw_data['image'][..., None], raw_data['label'].astype('int64')
        print(image_data.shape)
        train_size = int(0.8 * len(label_data))
        test_size = len(label_data) - train_size
        data = []
        for i in range(len(label_data)):
            data.append([np.array(image_data[i]), label_data[i]])
            
        random_seed = 2023 #可以改成随机种子
        np.random.seed(random_seed)
        np.random.shuffle(data)
        train_data = data[test_size:]  # 训练集
        test_data = data[:test_size]  # 测试集
        print("???")
        # 测试集的输入输出和训练集的输入输出
        X_train = np.array([i[0] for i in train_data])  # 训练集特征
        y_train = np.array([i[1] for i in train_data])  # 训练集标签
        X_test = np.array([i[0] for i in test_data])  # 测试集特征
        y_test = np.array([i[1] for i in test_data])  # 测试集标签

        np.savez('data/train.npz', image = X_train, label = y_train)
        np.savez('data/test.npz', image = X_test, label = y_test)
        print("done")
        train_dataset = CustomDataset(X_train, y_train, transform=transform_train)
        val_dataset = CustomDataset(X_test, y_test, transform=transform_test)
        '''
        test_data = np.load('/mnt/DataDrive/liyuze/work/tibet/SNN-RAT/data/test.npz')
        image_test, label_test = test_data['image'][..., None].squeeze(-1), test_data['label'].astype('int64')
        train_data = np.load('/mnt/DataDrive/liyuze/work/tibet/SNN-RAT/data/train.npz')
        image_train, label_train = train_data['image'][..., None].squeeze(-1), train_data['label'].astype('int64')
        train_dataset = CustomDataset(image_train, label_train, transform=transform_train)
        val_dataset = CustomDataset(image_test, label_test, transform=transform_test)


        norm = ((0.1307,), (0.3081,))
        return train_dataset, val_dataset, norm
