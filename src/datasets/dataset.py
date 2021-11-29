import os
from enum import Enum
from pathlib import Path
from sklearn.utils import check_random_state
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

from src.datasets.da_dataset import MultiDomainDatasets
from src.datasets.office31 import Office31Dataset
from src.datasets.sampler import SamplingConfig


def param_to_hash(param_dict):
    config_hash = hashlib.md5(
        json.dumps(param_dict, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return config_hash


class DatasetSizeType(Enum):
    Max = "max"  # size of the biggest dataset
    Source = "source"  # size of the source dataset

    @staticmethod
    def get_size(size_type, source_dataset, *other_datasets):
        if size_type is DatasetSizeType.Max:
            return max(list(map(len, other_datasets)) + [len(source_dataset)])
        elif size_type is DatasetSizeType.Source:
            return len(source_dataset)
        else:
            raise ValueError(
                f"Size type size must be 'max' or 'source', had '{size_type}'"
            )
            

class DatasetFactory:
    """
    This class takes a configuration dictionary and generates 
    a MultiDomainDataset class with the appropriate data.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_path = cfg.DATA.PATH
        self.download = cfg.DATA.DOWNLOAD
        self.n_fewshot = cfg.DATA.FEWSHOT
        self.short_name = cfg.DATA.SHORT_DATASET_NAME
        self.long_name = cfg.DATA.LONG_DATASET_NAME
        self.dataset_group = cfg.DATA.DATASET_GROUP
        self.weight_type = cfg.DATA.WEIGHT_TYPE
        self.size_type = cfg.DATA.SIZE_TYPE
        self.source = cfg.DATA.SOURCE
        self.target = cfg.DATA.TARGET
        
        os.makedirs(self.data_path, exist_ok=True)

    def is_semi_supervised(self):
        return self.n_fewshot is not None and self.n_fewshot > 0

    def get_multi_domain_dataset(self, random_state):
        self._create_dataset(random_state)
        return self.domain_datasets

    def get_data_args(self):
        """
        Returns dataset specific arguments necessary to build the network
        first returned item is number of classes. second is a tuple of 
        arguments to be passed to all network_factory functions.
        Returns:
            tuple: tuple containing:
                - int: the number of classes in the dataset
                - int or None: the input dimension
                - int or None: the number of channels for images
        """
        if self.dataset_group == "digits":
            return 10, 784, (self._num_channels,)
        if self.dataset_group == "office31":
            return 31, None, ()

    def get_data_short_name(self):
        return self.short_name

    def get_data_long_name(self):
        return self.long_name

    # def get_data_hash(self):
    #     return param_to_hash(self._data_config)

    def _create_dataset(self, random_state):
        random_state = check_random_state(random_state)
        if self.dataset_group == "digits":
            src, tgt = self._create_digits_access()
        elif self.dataset_group == "office31":
            src, tgt = self._create_office31_access()
        else:
            raise NotImplementedError(
                f"Unknown dataset type, you can need your own dataset here: {__file__}"
            )
        self._create_domain_dataset(src, tgt, random_state)

    def _create_domain_dataset(self, source_access, target_access, random_state):
        self.long_name = f"{self.long_name}_{self.weight_type}_{self.size_type}"

        if self.weight_type == "preset0":
            source_sampling_config = SamplingConfig(
                class_weights=np.arange(source_access.n_classes(), 0, -1)
            )
            target_sampling_config = SamplingConfig(
                class_weights=random_state.randint(1, 4, size=target_access.n_classes())
            )
        elif self.weight_type == "balanced":
            source_sampling_config = SamplingConfig(balance=True)
            target_sampling_config = SamplingConfig(balance=True)
        elif self.weight_type not in ["preset0", "balanced", "natural"]:
            raise ValueError(f"Unknown weighting method {self.weight_type}.")
        else:
            source_sampling_config = SamplingConfig()
            target_sampling_config = SamplingConfig()

        self.domain_datasets = MultiDomainDatasets(
            source_access=source_access,
            target_access=target_access,
            source_sampling_config=source_sampling_config,
            target_sampling_config=target_sampling_config,
            size_type=self.size_type,
            n_fewshot=self.n_fewshot,
        )
    
    def _create_digits_access(self):
        (
            source_access,
            target_access,
            self._num_channels,
        ) = digits.DigitDataset.get_accesses(
            digits.DigitDataset(self.source.upper()),
            digits.DigitDataset(self.target.upper()),
            data_path=self.data_path,
        )
        return source_access, target_access
    
    def _create_office31_access(self):
        source_access, target_access = Office31Dataset.get_accesses(
            Office31Dataset(self.source.lower()),
            Office31Dataset(self.target.lower()),
            data_path=self.data_path,
        )
        return source_access, target_access