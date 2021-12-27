import torch.nn as nn
import torchvision.models as models


def change_af_resnet(model_name='resnet50', pretrained=True):
    
    resnet = getattr(models, model_name)(pretrained=pretrained)
    
    # Set up an alternative activation function for ReLU.
    activation = nn.LeakyReLU(negative_slope=0.1,
                              inplace=True)
    
    resnet.relu = activation
    
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    for layer in layers:
        each_layer = getattr(resnet, layer)
        for l in each_layer.children():
            l.relu = activation
        
    return resnet