import torch
import torch.nn as nn
from torch.nn import functional as F


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    max_class = class_output.max(1)
    y_hat = max_class[1]
    correct = y_hat.eq(label.view(label.size(0)).type_as(y_hat))
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(
            class_output, label.type_as(y_hat).view(label.size(0))
        )
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return loss, correct


def compute_mmd(src, tgt, ker_mul=2.0, ker_num=5, fix_sig=None):
    
    n_samples = int(src.size()[0] + tgt.size()[0])
    total = torch.cat([src, tgt], dim=0)
    
    total_0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    total_1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    
    L2_dist = ((total_0 - total_1)**2).sum(2)
    
    if fix_sig:
        bandwidth = fix_sig
    else:
        bandwidth = torch.sum(L2_dist.data) / (n_samples*(n_samples-1))
        
    bandwidth /= ker_mul ** (ker_num//2)
    bandwidth_list = [bandwidth * (ker_mul*i) for i in range(ker_num)]
    
    ker_val = sum([torch.exp(-L2_dist / bw_temp) for bw_temp in bandwidth_list])
    
    batch_size = int(src.size()[0])
    
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += ker_val[s1, s2] + ker_val[t1, t2]
        loss -= ker_val[s1, t2] + ker_val[s2, t1]
    
    return loss / float(batch_size)