from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


def gaussian_kernel(x, y, sigmas):
    
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1 / (2*sigmas)
    
    # calculate pairwise distance
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    dist = torch.sum((x - y)**2, 1)
    dist = torch.transpose(dist, 0, 1).contiguous()
    
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)
    
    return torch.sum(torch.exp(-s), 0).view_as(dist)


def compute_mmd(src, tgt):
    
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10,
              15, 20, 25, 30, 35, 100,1e3, 1e4, 1e5, 1e6]
    
    gaussian_ker = partial(gaussian_kernel, 
                           sigmas=Variable(torch.cuda.FloatTensor(sigmas)))
    
    xx = torch.mean(gaussian_ker(src, src))
    xy = torch.mean(gaussian_ker(src, tgt))
    yy = torch.mean(gaussian_ker(tgt, tgt))
    
    return xx - 2*xy + yy
    