import os, sys
from os.path import join, dirname
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger

sys.path.append(join(dirname(__file__), "../.."))
from src.models.architecture import AdaptTrainer
from src.utils.factory import read_yaml, move_file, my_makedirs
from src.utils.callbacks import get_callbacks


def train(cfg, output_path):
    
    seed_everything(cfg.GENERAL.SEED)
    debug = cfg.GENERAL.DEBUG
    epochs = cfg.GENERAL.EPOCH
    
    csv_logger = CSVLogger(save_dir=str(output_path),
                           name=f'{cfg.MODEL.NAME}_{cfg.DATA.SHORT_DATASET_NAME}')
    wandb_logger = WandbLogger(project=f'{cfg.MODEL.NAME}_{cfg.DATA.SHORT_DATASET_NAME}')
    
    dirpath, checkpoint, early_stopping = get_callbacks(cfg, output_path)
    
    if early_stopping == False:
        callbacks = [checkpoint]
    else:
        callbacks = [checkpoint, early_stopping]

    trainer = Trainer(
        max_epochs=3 if debug else epochs,
        gpus=cfg.GENERAL.GPUS,
        strategy='dp',
        deterministic=False,
        benchmark=True,
        accumulate_grad_batches=1,
        callbacks=callbacks,
        logger=[csv_logger, wandb_logger]
    )
    
    model = AdaptTrainer(cfg)
    
    trainer.fit(model)
    results = trainer.test()
    
def main():
    
    cfg = read_yaml(fpath='src/configs/config.yaml')
    
    output_path = 'output'
    my_makedirs(output_path)
    
    train(cfg, output_path)
    
    move_file(cfg, output_path)
    
if __name__ == "__main__":
    main()
