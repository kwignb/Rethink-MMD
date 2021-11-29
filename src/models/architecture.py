import os, sys
from os.path import join, dirname

import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision.models as models
import pytorch_lightning as pl

sys.path.append(join(dirname(__file__), "../.."))
from src.models.losses import compute_mmd, cross_entropy_logits
from src.datasets.dataset import DatasetFactory
from src.utils.factory import freeze


def get_aggregated_metrics(metric_name_list, metric_outputs):
    metric_dict = {}
    for metric_name in metric_name_list:
        metric_dim = len(metric_outputs[0][metric_name].shape)
        if metric_dim == 0:
            metric_value = torch.stack(
                [x[metric_name] for x in metric_outputs]
            ).to(dtype=torch.float32).mean()
        else:
            metric_value = (
                torch.cat([x[metric_name] for x in metric_outputs]).double().mean()
            )
        metric_dict[metric_name] = metric_value.item()
    return metric_dict


def get_aggregated_metrics_from_dict(input_metric_dict):
    metric_dict = {}
    for metric_name, metric_value in input_metric_dict.items():
        metric_dim = len(metric_value.shape)
        if metric_dim == 0:
            metric_dict[metric_name] = metric_value
        else:
            metric_dict[metric_name] = metric_value.double().mean()
    return metric_dict


def get_metrics_from_parameter_dict(parameter_dict, device):
    return {k: torch.tensor(v, device=device) for k, v in parameter_dict.items()}


class AdaptTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model_name = cfg.MODEL.NAME
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.batch_size = cfg.PARAMS.BATCH_SIZE
        self._lambda = cfg.PARAMS.LAMBDA
        self.lr = cfg.PARAMS.LR
        self.freeze = cfg.GENERAL.FREEZE
        self.seed = cfg.GENERAL.SEED
        self.factory = DatasetFactory(cfg)
        self._dataset = self.factory.get_multi_domain_dataset(self.seed)
        self._dataset.prepare_data_loaders()
        self.__build_model()
        
    def __build_model(self):
        "Define model layers"
        
        # 1. Load pre-trained model
        model_func = getattr(models, self.model_name)
        backbone = model_func(pretrained=True)
        
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
        x = x.squeeze(-1).squeeze(-1)
        
        # 2. Classification
        class_output = self.fc(x)
        
        return x, class_output
        
    def compute_loss(self, batch, split_name='val'):
        
        (x_s, y_s), (x_t, y_t) = batch
        
        feat_s, yhat_s = self.forward(x_s)
        feat_t, yhat_t = self.forward(x_t)

        src_cls_loss, src_ok = cross_entropy_logits(yhat_s, y_s)
        tgt_cls_loss, tgt_ok = cross_entropy_logits(yhat_t, y_t)
        
        mmd_loss = compute_mmd(feat_s, feat_t)
        
        log_metrics = {f"{split_name}_src_acc": src_ok,
                       f"{split_name}_tgt_acc": tgt_ok,
                       f"{split_name}_mmd_loss": mmd_loss}

        return src_cls_loss, mmd_loss, log_metrics
    
    def train(self, mode=True):
        super(AdaptTrainer, self).train(mode=mode)
        if self.freeze:
            freeze(module=self.encoder, train_bn=False)
        
    def training_step(self, batch, batch_nb):
        
        cls_loss, mmd_loss, log_metrics = self.compute_loss(batch, split_name='tr')
        loss = cls_loss + self._lambda * mmd_loss
        
        log_metrics = get_aggregated_metrics_from_dict(log_metrics)

        log_metrics["tr_total_loss"] = loss
        log_metrics["tr_cls_loss"] = cls_loss
        
        return {"loss": loss,
                "log": log_metrics}
    
    def validation_step(self, batch, batch_nb):
        cls_loss, mmd_loss, log_metrics = self.compute_loss(batch)
        loss = cls_loss + self._lambda * mmd_loss
        log_metrics["val_loss"] = loss
        return log_metrics
    
    def _validation_epoch_end(self, outputs, metrics_at_valid):
        log_dict = get_aggregated_metrics(metrics_at_valid, outputs)
        device = outputs[0].get("val_loss").device

        avg_loss = log_dict["val_loss"]
        
        return {"val_loss": avg_loss,
                "log": log_dict}

    def validation_epoch_end(self, outputs):
        
        metrics_to_log = ("val_loss",
                          "val_src_acc",
                          "val_tgt_acc",
                          "val_mmd_loss")
        
        return self._validation_epoch_end(outputs, metrics_to_log)
    
    def test_step(self, batch, batch_nb):
        cls_loss, mmd_loss, log_metrics = self.compute_loss(batch, split_name="test")
        loss = cls_loss + self._lambda * mmd_loss
        log_metrics["test_loss"] = loss
        return log_metrics

    def test_epoch_end(self, outputs):
        metrics_at_test = ("test_loss",
                           "test_src_acc",
                           "test_tgt_acc",
                           "test_mmd_loss")
        
        log_dict = get_aggregated_metrics(metrics_at_test, outputs)

        return {"avg_test_loss": log_dict["test_loss"],
                "log": log_dict}
        
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