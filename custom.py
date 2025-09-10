import numpy as np
from PIL import Image
import torchvision
import torch
import os

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_local_data(root, n_labeled,
                      transform_train=None, transform_val=None):

    base_dataset = torchvision.datasets.ImageFolder(root=os.path.join(root,'train'), transform=transform_train)
    train_labeled_idxs, train_unlabeled_idxs = train_val_split(base_dataset.targets, int(n_labeled/4))

    train_labeled_dataset = Local_labeled(root=os.path.join(root, 'train'), indexs=train_labeled_idxs, transform=transform_train)
    train_unlabeled_dataset = Local_unlabeled(root=os.path.join(root, 'train'), indexs=train_unlabeled_idxs, transform=TransformTwice(transform_train))
    val_dataset = Local_labeled(root=os.path.join(root, 'val'), transform=transform_val)
    test_dataset = Local_labeled(root=os.path.join(root, 'test'), transform=transform_val)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset

def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(4):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs

Local_mean = (0.4914, 0.4822, 0.4465) 
Local_std = (0.2471, 0.2435, 0.2616) 

class Local_labeled(torchvision.datasets.ImageFolder):
    def __init__(self, root, indexs=None, transform=None, target_transform=None):
        super(Local_labeled, self).__init__(root=root, transform=transform)
        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        
class Local_unlabeled(Local_labeled):
    def __init__(self, root, indexs, transform=None, target_transform=None):
        super(Local_unlabeled, self).__init__(root=root, indexs=indexs, transform=transform, target_transform=target_transform)
        self.targets = np.array([-1 for i in range(len(self.targets))])