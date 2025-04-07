import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import random
import json
import wandb
import heapq
import re
from torch.nn.parallel import DistributedDataParallel as DDP
from base import BaseTrainer
from utils import MetricTracker, MetricTracker_scalars
from utils import transformsgpu
from utils import transformmasks
import torchvision.transforms.functional as trans_F
from models.loss import WBCELoss, KDLoss, ACLoss, MBCELoss, BCEWithLogitsLossWithIgnoreIndex,FeatureClusteringLoss,FeatureSeperationLoss,RSKnowledgeDistillationLoss
from torchvision.transforms.functional import InterpolationMode
from models.lovasz_losses import lovasz_softmax
from data_loader import VOC
import torch.distributed as dist

def strongTransform1(parameters, data=None, target=None, position=None, mask_n=None,width=None,height=None,num1=21):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.p_Mix_bg(mask=parameters["p_Mix_bg"], data=data, target=target, position=position, mask_n=mask_n)
    return data, target

def strongTransform2(parameters, data=None, target=None, position=None, mask_n=None,width=None,height=None,num1=21):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.Mix_up_bg(mask=parameters["Mix_up_bg"], data=data, target=target, num1=num1)
    return data, target

def strongTransform3(parameters, data=None, target=None, position=None, mask_nn=None,width=None,height=None,found_region = None,num1=21):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.p_Mix_beijing(mask=parameters["mix_beijing"], data=data, target=target, mask_nn=mask_nn, found_region = found_region, num1=num1)
    return data, target

def strongTransform4(parameters, data=None, target_0=None,target_1=None, mask_nn=None,found_region = None,num1=21):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.p_Mixup_beijing(mask=parameters["mixup_beijing"], data=data, target_0=target_0, target_1=target_1, mask_nn=mask_nn, found_region = found_region, num1=num1)
    return data, target


class Trainer_base(BaseTrainer):
    """
    Trainer class for a base step
    """
    def __init__(
        self, model, optimizer, evaluator, config, task_info,
        data_loader, lr_scheduler=None, logger=None, gpu=None,
    ):
        super().__init__(config, logger, gpu)
        if not torch.cuda.is_available():
            logger.info("using CPU, this will be slow")
        elif config['multiprocessing_distributed']:
            if gpu is not None:
                torch.cuda.set_device(self.device)
                model.to(self.device)
                # When using a single GPU per process and per
                # DDP, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
            else:
                model.to(self.device)
                # DDP will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = DDP(model)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = nn.DataParallel(model, device_ids=self.device_ids)

        self.optimizer = optimizer
        self.evaluator_val = evaluator[0]
        self.evaluator_test = evaluator[1]

        self.task_info = task_info
        self.n_old_classes = len(self.task_info['old_class'])  
        self.n_new_classes = len(self.task_info['new_class']) 
        self.num_classes = self.n_new_classes + self.n_old_classes+1   

        self.train_loader = data_loader[0]
        if self.train_loader is not None:
            self.len_epoch = len(self.train_loader)

        self.val_loader = data_loader[1]
        if self.val_loader is not None:
            self.do_validation = self.val_loader is not None

        self.test_loader = data_loader[2]
        if self.test_loader is not None:
            self.do_test = self.test_loader is not None
            
        self.lr_scheduler = lr_scheduler

        # For automatic mixed precision(AMP)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

        if self.evaluator_val is not None:
            self.metric_ftns_val = [getattr(self.evaluator_val, met) for met in config['metrics']]
        if self.evaluator_test is not None:
            self.metric_ftns_test = [getattr(self.evaluator_test, met) for met in config['metrics']]

        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce', 'loss_ac',
            writer=self.writer,
            colums=['total', 'counts', 'average'],
        )
        self.valid_metrics = MetricTracker_scalars(writer=self.writer)
        self.test_metrics = MetricTracker_scalars(writer=self.writer)

        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])


        self.method_vars = [
            'prototypes', 'count_features'
        ]

        self.prototypes = torch.zeros(
            (self.num_classes, self.config['proto_channels']),
            device=self.device,
            requires_grad=False
        ) 
        self.count_features = torch.zeros(
            (self.num_classes),
            device=self.device,
            requires_grad=False
        ) 
        pos_weight = torch.ones([len(self.task_info['new_class'])], device=self.device) * self.config['hyperparameter']['pos_weight']
        self.BCELoss = BCEWithLogitsLossWithIgnoreIndex(ignore_index=255, reduction='mean')
  
        self.ACLoss = ACLoss()
        self.loss_fc = FeatureClusteringLoss(factor=1.0, no_bkg=True)  
        self.loss_fs = FeatureSeperationLoss(factor=1.0, margin=self.config['margin'], num_class=self.num_classes, no_bkg=True)  

        self._print_train_info()

    def _print_train_info(self):
        self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
        self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce + {self.config['hyperparameter']['ac']} * L_ac + {self.config['hyperparameter']['loss_lovasz']} * L_lovasz + {self.config['hyperparameter']['fc']}*L_fc + {self.config['hyperparameter']['fs']}*L_fs" )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
        else:
            self.model.freeze_bn(affine_freeze=False)

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        layer = self.config['proto_layer']
        for batch_idx, data in enumerate(self.train_loader):
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                #15-1 step0:    logit:[8,16,512,512]  output:[8,256,32,32]
                logit,output = self.model(data['image'], ret_intermediate=False)
  
                if self.config['mix_mask'] == 'mixup_bg' or self.config['mix_mask'] == 'mixup_beijing':

                    num1 = self.n_old_classes+self.n_new_classes+1
                    n_cl = torch.tensor(num1).cuda()
                    labels_new_1 = torch.where(data['label'] != 255, data['label'], n_cl) 
                    if labels_new_1.dtype != torch.long:  
                        labels_new_1 = labels_new_1.long()
                    data['label'] = F.one_hot(labels_new_1, num1 + 1).float().permute(0, 3, 1, 2)
                    data['label'] = data['label'][:, :num1, :, :]  # remove 255 from 1hot   (2,12,513,513)

                loss_mbce = self.BCELoss(
                    logit,  # [N, |C1:t|, H, W]
                    data['label'],                   # [N, H, W]
                )

                loss_ac = self.ACLoss(logit[:, 0:1]).mean(dim=[0, 2, 3])  
                loss = self.config['hyperparameter']['mbce'] * loss_mbce + self.config['hyperparameter']['ac'] * loss_ac.sum() 

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
          
            self.train_metrics.update('loss_mbce', loss_mbce.item())   
            self.train_metrics.update('loss_ac', loss_ac.mean().item())
            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag

    def _valid_epoch(self, epoch):
        torch.distributed.barrier()
        
        log = {}
        self.evaluator_val.reset()
        self.logger.info(f"Number of val loader: {len(self.val_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, _ = self.model(data['image'])

                logit = torch.sigmoid(logit)
                pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]
                idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_val.add_batch(target, pred)

            if self.rank == 0:
                self.writer.set_step((epoch), 'valid')

            for met in self.metric_ftns_val:
                if len(met().keys()) > 2:
                    self.valid_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                else:
                    self.valid_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
                if 'by_class' in met().keys():
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in self.evaluator_val.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{VOC[i]} {met()['by_class'][i]:.2f}\n"
                        elif i in self.evaluator_val.old_classes_idx:
                            by_class_str = by_class_str + f"{i:2d}  {VOC[i]} {met()['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})
        return log



    def __update_protos(
        self, features, labels,
        prototypes, count_features
    ):
        """
        It takes features from current net and GT to update prototypes for each
        class. Note that only classes with GT are updated
        """
        device = features.device
        b, c, h, w = features.shape
 

        labels = trans_F.resize(labels, (h, w), InterpolationMode.NEAREST).unsqueeze(1)
        cls_new = torch.unique(labels) 

        if cls_new[0] == 0:
            cls_new = cls_new[1:]  
        if cls_new.shape[0] == 0:
            return
        if cls_new[-1] == 255:
            cls_new = cls_new[:-1]
        cls_new = cls_new.tolist()      
        features_cl_num = torch.zeros(self.num_classes, dtype=torch.long, device=device)         #15-1 step0: [15]
        features_cl_sum = torch.zeros((self.num_classes, c), dtype=torch.float, device=device)   #15-1 step0: [15,256]
        features_cl_mean = torch.zeros((self.num_classes, c), dtype=torch.float, device=device)  #15-1 step0: [15,256]
        features = features.permute(0, 2, 3, 1).reshape(-1, c)      #[8192,256]
        labels = labels.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()    #[8192]
        for cl in cls_new:  
            features_cl = features[(labels == cl), :]   
            features_cl_num[cl] = features_cl.shape[0] 
            features_cl_sum[cl] = torch.sum(features_cl, dim=0) #[256]
            features_cl_mean[cl] = torch.mean(features_cl, dim=0).to(torch.float32)   #[16,256]

        #if self.args['distributed']:
        if self.config['multiprocessing_distributed']:
            dist.all_reduce(features_cl_num, op=dist.ReduceOp.SUM)
            dist.all_reduce(features_cl_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(features_cl_mean, op=dist.ReduceOp.SUM)


        for cl in range(0, self.num_classes):  
            if features_cl_num[cl] <= 0:
                continue
            proto_running_mean = (features_cl_sum[cl] \
                + count_features[cl] * prototypes[cl]) \
                / (count_features[cl] + features_cl_num[cl])
            count_features[cl] += features_cl_num[cl]   
            prototypes[cl] = proto_running_mean       


    def _test(self, epoch=None):
        torch.distributed.barrier()

        log = {}
        self.evaluator_test.reset()
        self.logger.info(f"Number of test loader: {len(self.test_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, features = self.model(data['image'])
                logit = torch.sigmoid(logit)
                pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]

                idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]

                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_test.add_batch(target, pred)

            if epoch is not None:
                if self.rank == 0:
                    self.writer.set_step((epoch), 'test')

            for met in self.metric_ftns_test:
                if epoch is not None:
                    if len(met().keys()) > 2:
                        self.test_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                    else:
                        self.test_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
                if 'by_class' in met().keys():
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in self.evaluator_test.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{VOC[i]} {met()['by_class'][i]:.2f}\n"
                        else:
                            by_class_str = by_class_str + f"{i:2d}  {VOC[i]} {met()['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})
        return log


class Trainer_incremental(Trainer_base):
    """
    Trainer class for incremental steps
    """
    def __init__(
        self, model, model_old, optimizer, evaluator, config, task_info,
        data_loader,memory_set,lr_scheduler=None, logger=None, gpu=None,
    ):
        super().__init__(
            model=model, optimizer=optimizer, evaluator=evaluator, config=config, task_info=task_info,
            data_loader=data_loader,lr_scheduler=lr_scheduler, logger=logger, gpu=gpu)
        self.memory_set = memory_set
        if config['multiprocessing_distributed']:
            if gpu is not None:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old, device_ids=[gpu])
            else:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old)
        else:
            if model_old is not None:
                self.model_old = nn.DataParallel(model_old, device_ids=self.device_ids)
                
        memory_json = f"overlap_{self.config['data_loader']['args']['task']['name']}_EIR" + '/' + f"step_{self.config['data_loader']['args']['task']['step']}" + '/'+"memory.json"     #new baseline
        with open(memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        file_names = memory_list[f"step_{self.config['data_loader']['args']['task']['step']}"]["memory_list"]
        file_xinxi = memory_list[f"step_{self.config['data_loader']['args']['task']['step']}"]["memory_xinxi"]  
        file_curr_memorylist = memory_list[f"step_{self.config['data_loader']['args']['task']['step']}"]["curr_memory_list"] 

        self.memory_labelxinxi =  {i: [] for i in range(21+1)} 
        for la in file_curr_memorylist.keys():
            la1 = int(re.search(r'\d+', la).group())
            for file in file_curr_memorylist[la]:
                self.memory_labelxinxi[la1].append(file[0])
    #end
        self.logger.info(f"instance meomry xinxi: {self.memory_labelxinxi}")

        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce', 'loss_kd', 'loss_lovasz',
            writer=self.writer, colums=['total', 'counts', 'average'],
        )
        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        self.RSKDLoss = RSKnowledgeDistillationLoss(reduction='mean')     

    def _print_train_info(self):
        self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
        self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce + {self.config['hyperparameter']['kd']} * L_kd ")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
            self.model.module.freeze_dropout()
        else:
            self.model.freeze_bn(affine_freeze=False)
            self.model.freeze_dropout()
        self.model_old.eval()

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        
        layer = self.config['proto_layer']
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)  
          
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):

                if self.model_old is not None:
                    with torch.no_grad():
                        logit_old, features_old, _ = self.model_old(data['image'], ret_intermediate=True)
    
  
                pred_prob = torch.sigmoid(logit_old).detach()
                pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                pseudo_labels = torch.where(
                        (data['label'] <= 0) & (pred_labels > 0) & (pred_scores >= 0.7),
                        pred_labels,
                        data['label'])
                labels_copy = data['label'].clone().to(self.device, non_blocking=True)     
                data['label'] = pseudo_labels        

                if self.config['mix_mask'] == 'mixup_beijing':
                    num1 = self.n_old_classes+self.n_new_classes+1  
                    n_cl = torch.tensor(num1).cuda()
                    labels_new_1 = torch.where(data['label'] != 255, data['label'], n_cl) 
                    data['label'] = F.one_hot(labels_new_1, num1 + 1).float().permute(0, 3, 1, 2)
                    data['label'] = data['label'][:, :num1, :, :]  # remove 255 from 1hot   (2,12,513,513)                   
                    for image_i in range(data['image'].size(0)):
                        unique_classes_n = torch.unique(labels_copy[image_i]).tolist()      
                        unique_classes_p = torch.unique(pseudo_labels[image_i]).tolist()   

                        set_n = set(unique_classes_n)
                        set_p = set(unique_classes_p)
                        common_classes = list(set_n.intersection(set_p))    
                        different_classes_p = list(set_p.difference(set_n)) 

                        if len(different_classes_p) > 3:
                            different_classes_p = different_classes_p[:3]
    
                        unique_classes_y = torch.unique(pseudo_labels[image_i])        
                        classes_yt = unique_classes_y[(unique_classes_y != 0) & (unique_classes_y != 255)]
                        mask_n = transformmasks.generate_class_mask(pseudo_labels[image_i],classes_yt).cuda()        #[512,512]
                        dengfen_num = 250

                        boxes_with_zeros = []  
                        for i in range(0, mask_n.shape[0] - dengfen_num + 1, dengfen_num):
                            for j in range(0, mask_n.shape[0] - dengfen_num + 1, dengfen_num):
                                region = mask_n[i:i+dengfen_num, j:j+dengfen_num]
                                num_0 = (region == 0).sum().item()  
                                boxes_with_zeros.append(((i, j), num_0))

                        sorted_boxes = sorted(boxes_with_zeros, key=lambda x: x[1], reverse=True)[:3] 
                        i = 0
                        for class_p in different_classes_p:
                            found_region = sorted_boxes[i][0]
                            i = i+1           
                            #second method                        
                            randnum0 = random.randint(0, len(self.memory_labelxinxi[class_p])-1)   
                            i_name = self.memory_labelxinxi[class_p][randnum0]   #image name
                            
                            sample_total = self.memory_set.get_image_by_name(i_name)    #get image by image_name
                            m_img,m_lab,name6 = sample_total['image'],sample_total['label'], sample_total['image_name']
                            #check m_lab class
                            check = torch.unique(m_lab)    #check： class_p=1  check:tensor([0,1])                    
                            class_pt = torch.tensor(class_p).unsqueeze(0).to(self.device, dtype=torch.long, non_blocking=True)  
                            m_img = m_img.to(self.device, dtype=torch.float32, non_blocking=True)
                            m_lab = m_lab.to(self.device, dtype=torch.long, non_blocking=True)
                            MixMask = transformmasks.generate_class_mask(m_lab, class_pt).unsqueeze(0).cuda()   #[1,513,513]

                            strong_parameters = {"mixup_beijing": MixMask}     
                            data['image'][image_i], data['label'][image_i] = strongTransform4(strong_parameters,data=torch.cat((m_img.unsqueeze(0), data['image'][image_i].unsqueeze(0))), target_0 = m_lab, target_1 = data['label'][image_i], mask_nn = None,found_region=found_region, num1=num1)

                        if len(different_classes_p) < 2:
                            found_region = sorted_boxes[0][0]
                            inte = 3 - len(different_classes_p)
                            for i in range(0,inte):
                                randnum_class = random.randint(1, self.n_old_classes-1)  
                                randnum0 = random.randint(0, len(self.memory_labelxinxi[randnum_class])-1)  
                                i_name = self.memory_labelxinxi[randnum_class][randnum0]   
                                sample_total = self.memory_set.get_image_by_name(i_name)    
                                m_img,m_lab,name6 = sample_total['image'],sample_total['label'], sample_total['image_name']
                            
                                randnum_classt = torch.tensor(randnum_class).unsqueeze(0).to(self.device, dtype=torch.long, non_blocking=True)
                                m_img = m_img.to(self.device, dtype=torch.float32, non_blocking=True)
                                m_lab = m_lab.to(self.device, dtype=torch.long, non_blocking=True)
                                MixMask = transformmasks.generate_class_mask(m_lab, randnum_classt).unsqueeze(0).cuda() 
             
                                strong_parameters = {"mixup_beijing": MixMask}      
                                data['image'][image_i], data['label'][image_i] = strongTransform4(strong_parameters,data=torch.cat((m_img.unsqueeze(0), data['image'][image_i].unsqueeze(0))), target_0 = m_lab, target_1 = data['label'][image_i], mask_nn = None,found_region=found_region,num1=num1)

                logit, features, output = self.model(data['image'], ret_intermediate=True)  
                if self.model_old is not None:
                    with torch.no_grad():
                        logit_old, features_old,_ = self.model_old(data['image'], ret_intermediate=True)

                loss_mbce = self.BCELoss(
                    logit,  # [N, |C1:t|, H, W]
                    data['label'],                   # [N, H, W]
                )

                classes_yt = self.newclasses
                classes_ym = torch.tensor(classes_yt).unsqueeze(1).to(self.device, dtype=torch.long, non_blocking=True)     
                masknew = transformmasks.generate_class_mask(datalabel_copy,classes_ym).cuda()  #n*h*w

                num6 = self.n_old_classes+self.n_new_classes+1
                n_cl6 = torch.tensor(num6).cuda()
                labels_new_1 = torch.where(datalabel_copy != 255, datalabel_copy, n_cl6)  
                if not labels_new_1.dtype == torch.long:  
                    labels_new_1 = labels_new_1.long() 
                datalabel_copy = F.one_hot(labels_new_1, num6 + 1).float().permute(0, 3, 1, 2)
                datalabel_copy = datalabel_copy[:, :num6, :, :]  # remove 255 from 1hot   (2,12,513,513)
                masknew = datalabel_copy 
                maskn1 = 1-masknew      

                loss_kd = self.RSKDLoss(
                    logit, logit_old,mask=masknew
                )

                loss = self.config['hyperparameter']['mbce'] * loss_mbce + self.config['hyperparameter']['kd'] * loss_kd.sum()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())  
            self.train_metrics.update('loss_mbce', loss_mbce.item())           
            self.train_metrics.update('loss_kd', loss_kd.mean().item())

            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag