import math
import torch
import copy
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as trans_F
from torchvision.transforms.functional import InterpolationMode

class BCELoss(nn.Module):
    def __init__(self, ignore_index=255, ignore_bg=True, pos_weight=None, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.pos_weight = pos_weight
        self.reduction = reduction
        if ignore_bg is True: 
            self.ignore_indexes = [0, self.ignore_index]
        else:
            self.ignore_indexes = [self.ignore_index]
    def forward(self, logit, label, logit_old=None):
        # logit:     [N, C_tot, H, W]
        # logit_old: [N, C_prev, H, W]
        # label:     [N, H, W] or [N, C, H, W]
        C = logit.shape[1]
        if logit_old is None:
            if len(label.shape) == 3:
                # target: [N, C, H, W]
                target = torch.zeros_like(logit).float().to(logit.device)
                for cls_idx in label.unique():
                    if cls_idx in self.ignore_indexes:      
                        continue
                    target[:, int(cls_idx)] = (label == int(cls_idx)).float()   
                    #target[:, int(cls_idx)] = (label == int(cls_idx)).float()   
            elif len(label.shape) == 4:
                target = label
            else:
                raise NotImplementedError
            logit = logit.permute(0, 2, 3, 1).reshape(-1, C)
            target = target.permute(0, 2, 3, 1).reshape(-1, C)
            return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)(logit, target)
        else:
            if len(label.shape) == 3:
                #target: [N, C, H, W]
                target = torch.zeros_like(logit).float().to(logit.device)
                target[:, 1:logit_old.shape[1]] = logit_old.sigmoid()[:, 1:]
                for cls_idx in label.unique():
                    if cls_idx in self.ignore_indexes:
                        continue
                    target[:, int(cls_idx)] = (label == int(cls_idx)).float()
            else:
                raise NotImplementedError
            loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)(logit, target)
            del target

            return loss


class WBCELoss(nn.Module):
    def __init__(self, ignore_index=255, pos_weight=None, reduction='none', n_old_classes=0, n_new_classes=0):
        super().__init__()
        self.ignore_index = ignore_index    
        self.n_old_classes = n_old_classes  # |C0:t-1| + 1(bg), 19-1: 20 | 15-5: 16 | 15-1: 16...   
        self.n_new_classes = n_new_classes  # |Ct|, 19-1: 1 | 15-5: 5 | 15-1: 1
        
        self.reduction = reduction  
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=self.reduction)
    def forward(self, logit, label):

        N, C, H, W = logit.shape
        target = torch.zeros_like(logit, device=logit.device).float()   
        for cls_idx in label.unique():
            if cls_idx in [0, self.ignore_index]:
            #if cls_idx in [self.ignore_index]:
                continue
            #print(cls_idx)
            target[:, int(cls_idx) - self.n_old_classes] = (label == int(cls_idx)).float()  
        loss = self.criterion(
            logit.permute(0, 2, 3, 1).reshape(-1, C),
            target.permute(0, 2, 3, 1).reshape(-1, C)
        )
        if self.reduction == 'none':
            return loss.reshape(N, H, W, C).permute(0, 3, 1, 2)   #[N, C, H, W]
        elif self.reduction == 'mean':
            return loss
        else:
            raise NotImplementedError

class MBCELoss(nn.Module):
    def __init__(self, ignore_index=255, ignore_bg=True, pos_weight=None, reduction='none'):
        super().__init__()
        self.ignore_index = ignore_index
        self.pos_weight = pos_weight
        self.reduction = reduction
        if ignore_bg is True:
            self.ignore_indexes = [0, self.ignore_index]
        else:
            self.ignore_indexes = [self.ignore_index]
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=self.reduction)
    def forward(self, logit, label):
        N, C, H, W = logit.shape
        #print(logit.shape)
        target = torch.zeros_like(logit, device=logit.device).float() 
        for cls_idx in label.unique(): 
            if cls_idx in [0, self.ignore_index]:
                # if cls_idx in [self.ignore_index]:
                continue
            target[:, int(cls_idx)-1] = (label == int(cls_idx)).float()    
      
        loss = self.criterion(
            logit.permute(0, 2, 3, 1).reshape(-1, C),
            target.permute(0, 2, 3, 1).reshape(-1, C)
        )
       # print("error is not 4")
        if self.reduction == 'none':
            return loss.reshape(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]
            #print("error is not 1 here")
        elif self.reduction == 'mean':
            return loss
            #print("error is not 2 here")
        else:
            raise NotImplementedError

class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
     
        self.ignore_index = ignore_index 
    def forward(self, inputs, targets, weight=None):


        # n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        # labels_new = torch.where(targets != self.ignore_index, targets, n_cl)  
        # if not labels_new.dtype == torch.long:  
        #     labels_new = labels_new.long() 
        # targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)   #one_hot
        # targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot  
        
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        #loss has shape B*C*H*W 
        loss = loss.sum(dim=1)  # sum the contributions of the classes

        if weight is not None:
            loss = loss * weight
        if self.reduction == 'mean':        
            #return loss.mean()
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            #return loss.sum()
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            #return loss 
            return loss * targets.sum(dim=1)
        
class KDLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
    def forward(self, logit, logit_old=None):
        # logit:     [N, |Ct|, H, W]
        # logit_old: [N, |Ct|, H, W]
        N, C, H, W = logit.shape
        loss = self.criterion(
            logit.permute(0, 2, 3, 1).reshape(-1, C),
            logit_old.permute(0, 2, 3, 1).reshape(-1, C)
        ).reshape(N, H, W, C).permute(0, 3, 1, 2)   #[N,C,H,W]
        return loss

class IntraClassLoss(nn.Module):

    def __init__(self, factor: float=1.0):
        super().__init__()
        self.factor = factor

    def forward(
            self, features, features_old,
            outputs_old, labels, prototypes,
            num_old_class
        ):
        loss = torch.tensor(0., device=features.device)
        b, c, h, w = features.shape
        labels = trans_F.resize(labels, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        mask = (labels < num_old_class).long()
        pseudo = torch.argmax(outputs_old, dim=1)
        pseudo = trans_F.resize(pseudo, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        pseudo = pseudo * mask
        cls_old = torch.unique(pseudo)
        if cls_old[0] == 0:
            cls_old = cls_old[1:]

        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        features_old = features_old.permute(0, 2, 3, 1).reshape(-1, c)
        pseudo = pseudo.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()
        for cl in cls_old:
            features_cl = features[(pseudo == cl), :]              
            features_cl_old = features_old[(pseudo == cl), :]     
            prototype_cl = torch.mean(features_cl, dim=0).detach()  
            prototype_cl_old = torch.mean(features_cl_old, dim=0).detach() 

            criterion = nn.MSELoss(reduction='sum')
            loss_cl = criterion(
                features_cl - prototype_cl,
                features_cl_old - prototype_cl_old
            ) / features_cl.shape[0]

            loss += loss_cl

        if cls_old.shape[0] > 0:
            loss /= cls_old.shape[0]
            return self.factor * loss
        else:
            return torch.tensor(0., device=features.device)

class FeatureClusteringLoss(nn.Module):
    """
    Feature compacting loss in contrastive learning  

    Args:
        factor: The weight of this loss
        no_bkg: If ignore background class

    Inputs:
        features: Feature map of current network, with shape of B * C * H * W
        labels: GT of current input image, with shape of B * H * W
        prototypes: A list of prototypes for each class

    """

    def __init__(
            self, factor: float = 1.0, no_bkg: bool = False
    ):
        super().__init__()

        self.factor = factor
        self.no_bkg = no_bkg

    #features prototypes
    def forward(self, features, labels, prototypes):
        device = features.device
        b, c, h, w = features.shape
        labels = trans_F.resize(labels, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        cls_new = torch.unique(labels)      # the unique label in labels
        if self.no_bkg:
            cls_new = cls_new[1:]
        if cls_new[-1] == 255:              # cut 255 label
            cls_new = cls_new[:-1]

        loss = torch.tensor(0., device=device)  
        criterion = nn.MSELoss()                
        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        labels = labels.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()
        for cl in cls_new:
            cl = cl.item()
            features_cl = features[(labels == cl), :]   
            loss += criterion(
                features_cl,
                prototypes[cl].unsqueeze(0).expand(features_cl.shape[0], -1)
            )       
        loss /= cls_new.shape[0]   

        return self.factor * loss

class FeatureSeperationLoss(nn.Module):
    """
    Feature seperation part in contrastive learning  

    Args:
        factor: The weight of this loss
        margin: The minimal margin between class prototypes
        num_class: Number of classes
        no_bkg: If ignore background class

    Inputs:
        features: feature map of current network, with shape of B * C * H * W 
        labels: GT of current input image, with shape of B * H * W 
        prototypes: a list of prototypes for each class

    """

    def __init__(
            self, factor: float = 1.0, margin: float = 0., num_class: int = 0, no_bkg: bool = False
    ):
        super().__init__()

        self.factor = factor   
        self.margin = margin    
        self.no_bkg = no_bkg   
        self.num_class = num_class   

    def forward(self, features, labels, prototypes):
        device = features.device
        b, c, h, w = features.shape
        # Avoid changing prototypes
        prototypes = copy.deepcopy(prototypes)
        labels = trans_F.resize(labels, (h, w), InterpolationMode.NEAREST).unsqueeze(1)     
        cls_new = torch.unique(labels)  # the unique label in labels
        if self.no_bkg:                 # whether ingore the beijing class
            cls_new = cls_new[1:]
        if cls_new[-1] == 255:          # cut 255 label
            cls_new = cls_new[:-1]

        features_local_mean = torch.zeros((self.num_class, c), dtype=torch.float32, device=device)   # all 0 tensor to store every class feature mean 15-1 step0: [16,256]
        features = features.permute(0, 2, 3, 1).reshape(-1, c)  #[8192,226]
        labels = labels.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()    #[8192]
        for cl in cls_new:
            features_cl = features[(labels == cl), :]   #cl=3, [353,256]
            features_local_mean[cl] = torch.mean(features_cl, dim=0).to(torch.float32)          # calculate each class feature mean
        cls_new = cls_new.to(torch.long)  
 
        features_local_mean_reduced = features_local_mean[cls_new, :]           # select special class feature mean

        REF = features_local_mean_reduced
        REF = F.normalize(REF, p=2, dim=1)
        features_local_mean_reduced = F.normalize(features_local_mean_reduced, p=2, dim=1)     

        D = 1 - torch.mm(features_local_mean_reduced, REF.T)      
        for i in range(D.shape[0]):
            D[i][i] = 2.0
        loss = torch.mean(torch.clamp(self.margin - D, min=0.0)).to(torch.float32)    

        return self.factor * loss

class RSKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        new_cl = inputs.shape[1] - targets.shape[1]
        targets = targets * self.alpha

        new_bkg_idx = torch.arange(inputs.shape[1]).to(inputs.device)
        new_bkg_idx[1:targets.shape[1]+1] = 0 
        new_bkg_idx[targets.shape[1]+1:] = 1   
        den = torch.logsumexp(inputs, dim=1)
        outputs_no_bgk = inputs[:, 1:targets.shape[1]] - den.unsqueeze(dim=1)

        outputs_bkg = torch.logsumexp(inputs[:, new_bkg_idx == 1], dim=1) - den

        labels = torch.softmax(targets, dim=1)
        loss_no_bgk = (labels[:, 1:] * outputs_no_bgk).sum(dim=1)
        loss_bkg = labels[:, 0] * outputs_bkg
        loss = (loss_no_bgk + loss_bkg) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
                outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
                outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs
    
class FeatureClusteringLoss(nn.Module):
    """
        Feature compacting loss in contrastive learning
        Args:
            factor: The weight of this loss
            no_bkg: If ignore background class

        Inputs:
            features: Feature map of current network, with shape of B * C * H * W
            labels: GT of current input image, with shape of B * H * W
            prototypes: A list of prototypes for each class
        
    """
    def __init__(
        self, factor: float=1.0, no_bkg: bool=False
    ):
        super().__init__()

        self.factor = factor
        self.no_bkg = no_bkg

    def forward(self, features, labels, prototypes):
        device = features.device
        b, c, h, w = features.shape
        labels = trans_F.resize(labels, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        cls_new = torch.unique(labels)
        if self.no_bkg:
            cls_new = cls_new[1:]
        if cls_new[-1] == 255:
            cls_new = cls_new[:-1]

        loss = torch.tensor(0., device=device)
        criterion = nn.MSELoss()
        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        labels = labels.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()

        for cl in cls_new:
            features_cl = features[(labels == cl), :]
            loss += criterion(
                features_cl,
                prototypes[cl].unsqueeze(0).expand(features_cl.shape[0], -1)
            )
        
        loss /= cls_new.shape[0]

        return self.factor * loss
