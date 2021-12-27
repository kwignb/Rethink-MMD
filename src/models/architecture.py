import os, sys
from os.path import join, dirname

import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics import Accuracy, Recall, Precision, F1

sys.path.append(join(dirname(__file__), "../.."))
from src.models.losses import compute_mmd
from src.models.models import change_af_resnet
from src.datasets.dataset import DatasetFactory
from src.utils.factory import freeze


class AdaptTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super(AdaptTrainer, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model_name = cfg.MODEL.NAME
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.num_fewshot = cfg.DATA.FEWSHOT
        self.batch_size = cfg.PARAMS.BATCH_SIZE
        self._lambda = cfg.PARAMS.LAMBDA
        self.lr = cfg.PARAMS.LR
        self.freeze = cfg.GENERAL.FREEZE
        self.seed = cfg.GENERAL.SEED
        self.factory = DatasetFactory(cfg)
        self._dataset = self.factory.get_multi_domain_dataset(self.seed)
        self._dataset.prepare_data_loaders()
        self.__build_model()
        
        self.train_src_acc, self.train_tgt_acc = Accuracy(), Accuracy()
        self.val_src_acc, self.val_tgt_acc = Accuracy(), Accuracy()
        self.test_src_acc, self.test_tgt_acc = Accuracy(), Accuracy()
        self.test_src_F1 = F1(num_classes=self.num_classes, average='macro')
        self.test_tgt_F1 = F1(num_classes=self.num_classes, average='macro')
        
    def __build_model(self):
        "Define model layers"
        
        # 1. Load pre-trained model
        model_func = getattr(models, self.model_name)
        backbone = model_func(pretrained=True)
        # backbone = change_af_resnet(model_name=self.model_name)
        
        _layers = list(backbone.children())[:-3]
        self.encoder = nn.Sequential(*_layers)
        
        # Fine tuning or freeze "all" parameters
        if self.freeze:
            freeze(module=self.encoder, train_bn=False)

        _layers4 = list(backbone.children())[-3:-1]
        self.layer4 = nn.Sequential(*_layers4)

        # 2. Make classifier
        num_filters = backbone.fc.in_features
        self.fc = nn.Linear(num_filters, self.num_classes)
        
    def forward(self, x):
        
        # 1. Feature extraction
        x = self.encoder(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)

        # 2. Classification
        class_output = self.fc(x)
        
        return x, class_output
    
    def lossfun(self, pred, label):
        return F.cross_entropy(pred, label)
    
    def compute_eval(self, x, y):
        feat, yhat = self.forward(x)
        loss = self.lossfun(yhat, y)
        return feat, yhat, loss
        
    def train(self, mode=True):
        super(AdaptTrainer, self).train(mode=mode)
        if self.freeze:
            freeze(module=self.encoder, train_bn=False)
        
    def training_step(self, batch, batch_idx):
        
        (x_s, y_s), (x_t, y_t) = batch
        
        y_s, y_t = y_s.squeeze(), y_t.squeeze()
        
        feat_s, yhat_s, src_loss = self.compute_eval(x_s, y_s)
        feat_t, yhat_t, tgt_loss = self.compute_eval(x_t, y_t)

        mmd_loss = compute_mmd(feat_s, feat_t)
        
        loss = src_loss + self._lambda * mmd_loss
        if self.num_fewshot > 0:
            loss += tgt_loss
        
        self.log("train_mmd_loss", mmd_loss, prog_bar=True, logger=True)
        self.log("train_src_acc", self.train_src_acc(yhat_s, y_s), prog_bar=True, logger=True)
        self.log("train_tgt_acc", self.train_tgt_acc(yhat_t, y_t), prog_bar=True, logger=True)
        
        return {"loss": loss,
                "train_mmd_loss": mmd_loss.detach(),
                "train_src_acc": self.train_src_acc(yhat_s, y_s).detach(),
                "train_tgt_acc": self.train_tgt_acc(yhat_t, y_t).detach()}
    
    def validation_step(self, batch, batch_idx):
        
        (x_s, y_s), (x_t, y_t) = batch
        
        y_s, y_t = y_s.squeeze(), y_t.squeeze()
        
        feat_s, yhat_s, src_loss = self.compute_eval(x_s, y_s)
        feat_t, yhat_t, tgt_loss = self.compute_eval(x_t, y_t)
        
        mmd_loss = compute_mmd(feat_s, feat_t)
        
        loss = src_loss + self._lambda * mmd_loss
        if self.num_fewshot > 0:
            loss += tgt_loss
        
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_mmd_loss", mmd_loss, prog_bar=True, logger=True)
        self.log("val_src_acc", self.val_src_acc(yhat_s, y_s), prog_bar=True, logger=True)
        self.log("val_tgt_acc", self.val_tgt_acc(yhat_t, y_t), prog_bar=True, logger=True)
        
        return {"val_loss": loss,
                "val_mmd_loss": mmd_loss,
                "val_src_acc": self.val_src_acc(yhat_s, y_s),
                "val_tgt_acc": self.val_tgt_acc(yhat_t, y_t)}
    
    def test_step(self, batch, batch_idx):
        
        (x_s, y_s), (x_t, y_t) = batch
        
        y_s, y_t = y_s.squeeze(), y_t.squeeze()
        
        feat_s, yhat_s, src_loss = self.compute_eval(x_s, y_s)
        feat_t, yhat_t, tgt_loss = self.compute_eval(x_t, y_t)
        
        mmd_loss = compute_mmd(feat_s, feat_t)
        
        loss = src_loss + self._lambda * mmd_loss
        if self.num_fewshot > 0:
            loss += tgt_loss
        
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_mmd_loss", mmd_loss, prog_bar=True, logger=True)
        self.log("test_src_acc", self.test_src_acc(yhat_s, y_s), prog_bar=True, logger=True)
        self.log("test_tgt_acc", self.test_tgt_acc(yhat_t, y_t), prog_bar=True, logger=True)
        self.log("test_src_F1", self.test_src_F1(yhat_s, y_s), prog_bar=True, logger=True)
        self.log("test_tgt_F1", self.test_tgt_F1(yhat_t, y_t), prog_bar=True, logger=True)
        
        return {"test_loss": loss,
                "test_mmd_loss": mmd_loss,
                "test_src_acc": self.test_src_acc(yhat_s, y_s),
                "test_tgt_acc": self.test_tgt_acc(yhat_t, y_t),
                "test_src_F1": self.test_src_F1(yhat_s, y_s),
                "test_tgt_F1": self.test_tgt_F1(yhat_t, y_t)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mmd_loss = torch.stack([x['val_mmd_loss'] for x in outputs]).mean()
        avg_src_acc = torch.stack([x['val_src_acc'] for x in outputs]).mean()
        avg_tgt_acc = torch.stack([x['val_tgt_acc'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss,
                'avg_val_mmd_loss': avg_mmd_loss,
                'avg_val_src_acc': avg_src_acc,
                'avg_val_tgt_acc': avg_tgt_acc}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_mmd_loss = torch.stack([x['test_mmd_loss'] for x in outputs]).mean()
        avg_src_acc = torch.stack([x['test_src_acc'] for x in outputs]).mean()
        avg_tgt_acc = torch.stack([x['test_tgt_acc'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss,
                'avg_test_mmd_loss': avg_mmd_loss,
                'avg_test_src_acc': avg_src_acc,
                'avg_test_tgt_acc': avg_tgt_acc}
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {'scheduler': optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=5,
                                                            gamma=0.2)}
        return [optimizer], [scheduler]
        
    def train_dataloader(self):
        dataloader = self._dataset.get_domain_loaders(
            split="train", batch_size=self.batch_size
        )
        return dataloader
    
    def val_dataloader(self):
        dataloader = self._dataset.get_domain_loaders(
            split="valid", batch_size=self.batch_size
        )
        return dataloader
    
    def test_dataloader(self):
        dataloader = self._dataset.get_domain_loaders(
            split="test", batch_size=self.batch_size
        )
        return dataloader