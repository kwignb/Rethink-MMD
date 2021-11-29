import os, sys
from os.path import join, dirname
from glob import glob

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.append(join(dirname(__file__), "../.."))
from src.models.architecture import AdaptTrainer


def get_callbacks(cfg, output_path):
    
    filename = f'{cfg.MODEL.NAME}_{cfg.DATA.SHORT_DATASET_NAME}'
    dirpath = str(output_path) + '/' + filename
    
    checkpoint = ModelCheckpoint(
        dirpath=dirpath, 
        filename=filename,
        **cfg.CALLBACKS.MODEL_CHECKPOINT.PARAMS
    )
    
    print(f'Early stopping: {cfg.CALLBACKS.EARLY_STOPPING.FLAG}')  
    if cfg.CALLBACKS.EARLY_STOPPING.FLAG:
        early_stopping = EarlyStopping(**cfg.CALLBACKS.EARLY_STOPPING.PARAMS)
    else:
        early_stopping = False
        
    return dirpath, checkpoint, early_stopping


def save_model(cfg, dirpath):
    
    ckpt_path = glob(dirpath + '/*.ckpt')
    
    if len(ckpt_path) == 1:
        ckpt_path = ckpt_path[0]
    else:
        print(f'There are more than one weight file found : {ckpt_path}')
    
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['state_dict']
    
    for key in list(state_dict.keys()):
        if key.split('.')[0] == 'encoder':
            state_dict[key.split('encoder.')[1]] = state_dict[key]
        elif key.split('.')[0] == 'fc':
            state_dict[key.split('fc.')[1]] = state_dict[key]
        del state_dict[key]
        
    ckpt_model_name = ckpt_path.replace('ckpt', 'pth').split('/')[-1]
    
    version_path = sorted(glob(dirpath + '/version_*'))[-1]
    torch.save(state_dict, version_path + '/' + ckpt_model_name)
    
    os.remove(ckpt_path)
