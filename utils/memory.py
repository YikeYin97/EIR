"""
"""

import math
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import models.model as module_arch
import utils.metric as module_metric
import utils.lr_scheduler as module_lr_scheduler
import data_loader.data_loaders as module_data

from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from scipy import ndimage

from data_loader.task import get_task_labels, get_per_task_classes
from trainer.trainer_voc import Trainer_base, Trainer_incremental


def _prepare_device(n_gpu_use, logger):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine,"
                       "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                       "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def memory_sampling_balanced(config, model, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.to(device)
            model = DDP(model, device_ids=[gpu])
        else:
            model.to(device)

            model = DDP(model)
    else:
       
        model = nn.DataParallel(model, device_ids=device_ids)
    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)      
    prev_num_classes = len(old_classes)  # 15
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'   
    memory_size = config['data_loader']['args']['memory']['mem_size']
    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file) 
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]   #[image, label]
    else:
        memory_list = {}
        memory_candidates = []

    logger.info("...start memory candidates collection")
    torch.distributed.barrier() 
    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():   
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
        else:
            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            image = images[b]
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()  
            if 0 in labels:
                labels.remove(0)
            choose_or_not = liantong_choose(target, labels)
            if choose_or_not:
                memory_candidates.append([img_name, labels])   
            else:
                continue
            memory_candidates.append([img_name, labels])
        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")         

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  
    sorted_memory_candidates = memory_candidates.copy()    
    np.random.shuffle(sorted_memory_candidates)
    
    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)
    num_sampled = 0
    
    while memory_size > num_sampled:
        for cls in random_class_order:
            #
            for idx, mem in enumerate(sorted_memory_candidates):      
                img_name, labels = mem                  
                if cls in labels:
                    curr_memory_list[f"class_{cls}"].append(mem)
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break
                    
            if memory_size <= num_sampled:
                break
        
    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list]),
        "memory_xinxi": {mem[0]: mem[1:] for mem in sampled_memory_list}
    }

    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()

def memory_sampling_balanced1(config, model, train_loader, task_info, logger, gpu):
    if gpu is None:
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.to(device)
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.to(device)
            model = DDP(model)
    else:
        model = nn.DataParallel(model, device_ids=device_ids)
    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step) 
    prev_num_classes = len(old_classes)  # 15
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'   
    memory_size = config['data_loader']['args']['memory']['mem_size']
    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file) 
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]   #[image, label]
    else:
        memory_list = {}
        memory_candidates = []

    logger.info("...start memory candidates collection")
    torch.distributed.barrier() 
    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():   
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']     
                true_targets = data['true_label'].to(device)       
        else:
            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
            true_targets = data['true_label'].to(device)

        jiexian = 1
        for b in range(images.size(0)):
            image = images[b]
            img_name = img_names[b]
            target = targets[b]
            true_target = true_targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()  
            true_labels = torch.unique(true_target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)

            if 0 in true_labels:
                true_labels.remove(0)


            if (len(true_labels) - len(labels)) > jiexian:
                memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")       

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15       
    sorted_memory_candidates = memory_candidates.copy()     
    np.random.shuffle(sorted_memory_candidates)              
    
    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)
    num_sampled = 0
    while memory_size > num_sampled:
        for cls in random_class_order:
            for idx, mem in enumerate(sorted_memory_candidates):      
                img_name, labels = mem    
                if cls in labels: 
                    curr_memory_list[f"class_{cls}"].append(mem)       
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break
                    
            if memory_size <= num_sampled:
                break
        
    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]         # gather all memory   curr_memory_list

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list]),
        "memory_xinxi": {mem[0]: mem[1:] for mem in sampled_memory_list}
    }

    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()

def instance_sampling_balanced(config, model, train_loader, task_info, logger, gpu):
    if gpu is None:
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.to(device)
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.to(device)
            model = DDP(model)
    else:
        model = nn.DataParallel(model, device_ids=device_ids)
    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step) 
    prev_num_classes = len(old_classes)  # 15

    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'   
    memory_size = config['data_loader']['args']['memory']['mem_size']
    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file) 
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]   #[image, label]
    else:
        memory_list = {}
        memory_candidates = []

    logger.info("...start memory candidates collection")
    torch.distributed.barrier() 
    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():   
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']     

        else:
            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']


        for b in range(images.size(0)):
            image = images[b]
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()  
            if 0 in labels:
                labels.remove(0)

            choose_or_not = liantong_choose(target, labels)
            if choose_or_not:
                memory_candidates.append([img_name, labels])   
            else:
                continue

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")      

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)} 
    sorted_memory_candidates = memory_candidates.copy()    
    np.random.shuffle(sorted_memory_candidates)
    
    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)       
    num_sampled = 0
    
    while memory_size > num_sampled:
        for cls in random_class_order:
            for idx, mem in enumerate(sorted_memory_candidates):      
                img_name, labels = mem    
                if cls in labels:
                    curr_memory_list[f"class_{cls}"].append(mem)       
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break
                    
            if memory_size <= num_sampled:
                break
        
    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]         # gather all memory

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list]),
        "memory_xinxi": {mem[0]: mem[1:] for mem in sampled_memory_list},
        "curr_memory_list":curr_memory_list
    }
    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()


def liantong_choose(target, labels):
    liantong = True
    for label in labels:
        if label == 0 or label == 255:      
            continue
        if not liantong1(target,label):     
            liantong = False
            break      
    return liantong


def liantong1(target, label):
    target_np = target.cpu().numpy()
    labeled_image, num_features = ndimage.label(target_np == label)
    if num_features == 1:
        return True   
    else:
        return False  



def custom_condition(target, label, edge_margin=3):

    near_edge = is_class_near_edge(target, label, edge_margin)
    class_area = calculate_class_area(target, label)
    total_area = calculate_total_area(target)
    if (not near_edge):
        return True
    else:
        return False
    
def is_class_near_edge(target, label, edge_margin=3):
    target_class_pixels = np.where(target == label)
    if len(target_class_pixels[0]) == 0:
        return False  
    min_row, min_col = np.min(target_class_pixels, axis=1)
    max_row, max_col = np.max(target_class_pixels, axis=1)

    is_near_edge = (
        min_row < edge_margin or
        min_col < edge_margin or
        max_row >= target.shape[0] - edge_margin or
        max_col >= target.shape[1] - edge_margin
    )
    return is_near_edge

def calculate_class_area(target,label):
    class_pixels = (label == target)
    class_area = class_pixels.sum()
    class_area_scalar = class_area.item()
    return class_area_scalar

def calculate_total_area(label):
    total_area = label.size(0)*label.size(1)
    return total_area  