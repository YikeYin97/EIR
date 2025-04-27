# Beyond Background Shift: Rethinking Instance Replay in Continual Semantic Segmentation
This is an official implementation of the paper "Beyond Background Shift: Rethinking Instance Replay in Continual Semantic Segmentation", accepted to CVPR 2025.
Our paper will be found at https://arxiv.org/abs/2503.22136.
![figure](https://github.com/user-attachments/assets/448257b2-517b-477f-9ed4-45efbc1301c1)

## Abstract
In this work, we focus on continual semantic segmentation (CSS), where segmentation networks are required to continuously learn new classes without erasing knowledge of previously learned ones. Although storing im
ages of old classes and directly incorporating them into the training of new models has proven effective in mitigat
ing catastrophic forgetting in classification tasks, this strategy presents notable limitations in CSS. Specifically, the stored and new images with partial category annotations leads to confusion between unannotated categories and the background, complicating model fitting. To tackle this issue, this paper proposes a novel Enhanced Instance Replay (EIR) method, which not only preserves knowledge of old classes while simultaneously eliminating background confusion by instance storage of old classes, but also mitigates background shifts in the new images by integrating stored instances with new images. By effectively resolving background shifts in both stored and new images, EIR alleviates catastrophic forgetting in the CSS task, thereby enhancing the model’s capacity for CSS. Experimental results validate the efficacy of our approach, which significantly outperforms state-of-the-art CSS methods.

## Pre-requisites
Python (3.8)  
Pytorch (1.12.1)  
numpy  
pillow  
scikit-learn  
tqdm  
matplotlib  

## Training

### An example srcipt for 5-3 overlapped setting of PASCAL VOC

```
PORT='tcp://127.0.0.1:24568'
GPU=1
BS=32  # Total 32
SAVEDIR='_saved_voc'   
TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='5-3'
EPOCH=60
INIT_LR=0.001
LR=0.0001
INIT_POSWEIGHT=2
MEMORY_SIZE=100 
NAME='EIR'

CUDA_VISIBLE_DEVICES=1 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --pos_weight ${INIT_POSWEIGHT}

CUDA_VISIBLE_DEVICES=1 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

CUDA_VISIBLE_DEVICES=1 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

CUDA_VISIBLE_DEVICES=1 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

CUDA_VISIBLE_DEVICES=1 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

CUDA_VISIBLE_DEVICES=1 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
```

## Testing
```
CUDA_VISIBLE_DEVICES=1 python eval_voc.py -d 0 --test -r ${SAVEDIR}/models/${TASKSETTING}_${TASKNAME}_${NAME}/step_5/checkpoint-epoch${EPOCH}.pth
```
