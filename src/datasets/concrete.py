import os, tarfile, logging, urllib, glob, requests
import urllib.parse
from PIL import Image
from enum import Enum

import numpy as np
from sklearn import preprocessing

import torch
import torch.utils.data as data

import src.datasets.transforms as proc


class DatasetAccess:
    """
    This class ensures a unique API is used to access training, validation and test splits
    of any dataset.
    """

    def __init__(self, n_classes):
        self._n_classes = n_classes

    def n_classes(self):
        return self._n_classes

    def get_train(self):
        """
        returns: a torch.utils.data.Dataset
        """
        raise NotImplementedError()

    def get_train_val(self, val_ratio):
        train_dataset = self.get_train()
        ntotal = len(train_dataset)
        ntrain = int((1 - val_ratio) * ntotal)
        torch.manual_seed(torch.initial_seed())
        return torch.utils.data.random_split(train_dataset, [ntrain, ntotal - ntrain])

    def get_test(self):
        raise NotImplementedError()


class ConcreteDatasetAccess(DatasetAccess):
    def __init__(self, domain, data_path):
        super().__init__(n_classes=2)
        self._data_path = data_path
        self._transform = proc.get_transform("concrete")
        self._domain = domain.value

    def get_train(self):
        return Concrete(self._data_path,
                        domain=self._domain,
                        train=True,
                        transform=self._transform)

    def get_test(self):
        return Concrete(self._data_path,
                        domain=self._domain,
                        train=False,
                        transform=self._transform)


class ConcreteDataset(Enum):
    CCIC = "ccic"
    SDNET = "sdnet"
    
    @staticmethod
    def get_accesses(source: "ConcreteDataset",
                     target: "ConcreteDataset",
                     data_path):

        return (ConcreteDatasetAccess(source, data_path),
                ConcreteDatasetAccess(target, data_path))


class Concrete(data.Dataset):
    """
    Concrete Domain Adaptation Dataset
    Args:
        root (string):
            Root directory of dataset where dataset file exist.
        train (bool, optional):
            If True, resample from dataset randomly.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, domain=None, train=True, transform=None):
        """
        Init Concrete dataset.
        """
        # init params
        self.root = os.path.expanduser(root)
        self.dirname = "Concrete"
        self.train = train

        self.transform = transform
        self.dataset_size = None
        self.domain = domain

        self.labeler = preprocessing.LabelEncoder()
        self.data, self.targets = self.load_samples()
        self.targets = torch.LongTensor(self.targets)

    def __getitem__(self, index):
        """
        Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path = self.data[index]
        img = None
        with open(path, "rb") as f:
            with Image.open(f) as imgf:
                img = imgf.convert("RGB")

        label = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        return img, label

    def __len__(self):
        """
        Return size of dataset.
        """
        return self.dataset_size

    def load_samples(self):
        """
        Load sample images from dataset.
        """
        imgdir = os.path.join(self.root, self.dirname, self.domain)
        if self.train:
            image_list = glob.glob(f"{imgdir}/train/*/*.jpg")
        else:
            image_list = glob.glob(f"{imgdir}/test/*/*.jpg")
        
        if len(image_list) == 0:
            raise RuntimeError("Concrete dataset is empty.")
        
        labels = [os.path.split(os.path.split(p)[0])[-1] for p in image_list]
        labels = self.labeler.fit_transform(labels)
        
        images = np.array(image_list).tolist()
        labels = np.array(labels).tolist()
        
        self.dataset_size = len(images)
        if self.train:
            self.labeler
            
        return images, labels