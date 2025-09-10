import logging
import math
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .randaugment import RandAugmentMC
logger = logging.getLogger(__name__)


custom_mean = (0.5, 0.5, 0.5)
custom_std = (0.5, 0.5, 0.5)

class ThreeClassDataset(Dataset):
    def __init__(self, root, indices=None, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.data = []
        self.targets = []
        self.class_to_idx = self._find_classes(self.root)

        # Load data
        for label in os.listdir(root):
            class_dir = os.path.join(root, label)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_file)
                        self.data.append(img_path)
                        self.targets.append(self.class_to_idx[label])

        if indices is not None:
            self.data = [self.data[i] for i in indices]
            self.targets = [self.targets[i] for i in indices]

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return class_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_custom_dataset(args, root, root1):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=custom_mean, std=custom_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=custom_mean, std=custom_std)
    ])

    base_dataset = ThreeClassDataset(root, transform=transform_labeled)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = ThreeClassDataset(root, indices=train_labeled_idxs, transform=transform_labeled)
    train_unlabeled_dataset = ThreeClassDataset(root, indices=train_unlabeled_idxs, transform=TransformFixMatch(mean=custom_mean, std=custom_std))

    test_dataset = ThreeClassDataset(root1, train=False, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224))
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224)),
            RandAugmentMC(n=2, m=10)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

DATASET_GETTERS = {'custom': get_custom_dataset}


