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


class Office31DatasetAccess(DatasetAccess):
    def __init__(self, domain, data_path):
        super().__init__(n_classes=31)
        self._data_path = data_path
        self._transform = proc.get_transform("office")
        self._domain = domain.value

    def get_train(self):
        return Office31(
            self._data_path,
            domain=self._domain,
            train=True,
            transform=self._transform,
            download=True,
        )

    def get_test(self):
        return Office31(
            self._data_path,
            domain=self._domain,
            train=False,
            transform=self._transform,
            download=True,
        )


class Office31Dataset(Enum):
    Amazon = "amazon"
    DSLR = "dslr"
    Webcam = "webcam"

    @staticmethod
    def get_accesses(source: "Office31Dataset", target: "Office31Dataset", data_path):

        return (
            Office31DatasetAccess(source, data_path),
            Office31DatasetAccess(target, data_path),
        )


class Office31(data.Dataset):
    """
    Office31 Domain Adaptation Dataset from the
    `Domain Adaptation Project at Berkeley 
    <https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code>`_.
    Args:
        root (string):
            Root directory of dataset where dataset file exist.
        train (bool, optional):
            If True, resample from dataset randomly.
        download (bool, optional):
            If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://docs.google.com/uc?export=download&id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE"

    def __init__(self, root, domain=None, train=True, transform=None, download=False):
        """Init Office31 dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "domain_adaptation_images.tar.gz"
        self.dirname = "Office31"
        self.train = train

        self.transform = transform
        self.dataset_size = None
        self.domain = domain

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        self.labeler = preprocessing.LabelEncoder()
        self.data, self.targets = self.load_samples()
        self.targets = torch.LongTensor(self.targets)

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # path = os.path.join(self.root, self.dirname, self.domain, self.data[index])
        path = self.data[index]
        img = None
        with open(path, "rb") as f:
            with Image.open(f) as imgf:
                img = imgf.convert("RGB")

        label = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.join(self.root, self.dirname)
        if not os.path.exists(filename):
            logging.info("Downloading " + self.url)
            with requests.Session() as session:
                resp = session.head(self.url)
                confirm = None
                for key, value in resp.cookies.items():
                    if "download_warning" in key:
                        confirm = value
                        break
                if confirm is None:
                    raise RuntimeError("Could not find 'download_warning' in cookies")
                resp = session.get(f"{self.url}&confirm={urllib.parse.quote(confirm)}")
                with open(filename, "wb") as f:
                    f.write(resp.content)
            os.makedirs(dirname, exist_ok=True)
            logging.info("Extracting files to " + dirname)
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(path=dirname)
        logging.info("[DONE]")

    def load_samples(self):
        """Load sample images from dataset."""
        imgdir = os.path.join(self.root, self.dirname, self.domain, "images")
        image_list = glob.glob(f"{imgdir}/*/*.jpg")
        if len(image_list) == 0:
            raise RuntimeError("Offce31 dataset is empty. Maybe it was not downloaded.")
        labels = [os.path.split(os.path.split(p)[0])[-1] for p in image_list]
        labels = self.labeler.fit_transform(labels)
        n_total = len(image_list)
        n_test = int(0.1 * n_total)
        indices = np.arange(n_total)
        rg = np.random.RandomState(seed=128753)
        rg.shuffle(indices)
        train_indices = indices[:-n_test]
        test_indices = indices[-n_test:]
        if self.train:
            images = np.array(image_list)[train_indices].tolist()
            labels = np.array(labels)[train_indices].tolist()
            self.dataset_size = len(images)
            self.labeler
        else:
            images = np.array(image_list)[test_indices].tolist()
            labels = np.array(labels)[test_indices].tolist()
            self.dataset_size = len(images)
        return images, labels