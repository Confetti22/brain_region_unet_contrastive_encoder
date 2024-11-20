import torch
import torch.nn as nn
from info_nec_loss import _info_nce_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.contrastive_loss=_info_nce_loss
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, preds, labels=None):
         
        logits, labels =  self.contrastive_loss(self.args,preds,labels)
        loss = self.loss_fn(logits, labels)

        return loss


def get_loss(args):
	
    return ContrastiveLoss(args)
