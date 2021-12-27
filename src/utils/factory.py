import os, glob, shutil
from omegaconf import OmegaConf
import torch

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


def read_yaml(fpath='configs/config.yaml'):
    config = OmegaConf.load(fpath)
    return config


def move_file(cfg, op_path):
    
    model_path = f'{cfg.MODEL.NAME}_{cfg.DATA.SHORT_DATASET_NAME}'
    from_path = op_path + '/' + model_path + '/' + model_path + '.ckpt'
    all_version_path = sorted(glob.glob(op_path + '/' + model_path + '/version_*'))
    
    num_list = []
    for v_path in all_version_path:
        version = v_path.split('/')[-1]
        num_list.append(int(version.split('_')[-1]))
    version_num = str(max(num_list))
    
    to_path = op_path + '/' + model_path + '/version_' + version_num
    shutil.move(from_path, to_path)

    
def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
        
def _make_trainable(module):
    """Unfreeze a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()
    

def _recursive_freeze(module, train_bn=True):
    """Freeze the layers of a given module.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
        Otherwise, they will be set to eval mode along with the other modules.
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module, n=-1, train_bn=True):
    """Freeze the layers up to index n.
    Operates in-place.
    Parameters
    ----------
    module : instance of `torch.nn.Module`
    n : int
        By default, all the layers will be frozen. Otherwise, an integer
        between 0 and `len(module.children())` must be given.
    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
    """
    idx = 0
    children = list(module.children())
    n_max = len(children) if n == -1 else int(n)
    for child in children:
        if idx < n_max:
            _recursive_freeze(module=child, train_bn=train_bn)
        else:
            _make_trainable(module=child)