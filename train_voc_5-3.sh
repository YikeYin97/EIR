#!/bin/bash
PORT='tcp://127.0.0.1:29557'
GPU=0
BS=24  # Total 32
SAVEDIR='region_n/'   

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='5-3'
EPOCH=60
INIT_LR=0.001
LR=0.0001
INIT_POSWEIGHT=2
MEMORY_SIZE=100

NAME='EIR'

CUDA_VISIBLE_DEVICES=0 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --pos_weight ${INIT_POSWEIGHT}

CUDA_VISIBLE_DEVICES=0 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

CUDA_VISIBLE_DEVICES=0 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

CUDA_VISIBLE_DEVICES=0 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

CUDA_VISIBLE_DEVICES=0 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

CUDA_VISIBLE_DEVICES=0 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

