#!/bin/bash

PORT='tcp://127.0.0.1:12346'
GPU=7
BS=32  # Total 32
SAVEDIR='noreplay_addpseudo'       

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='15-5'
EPOCH=60
INIT_LR=0.001
LR=0.0001
INIT_POSWEIGHT=2
MEMORY_SIZE=0 

NAME='EIR'

CUDA_VISIBLE_DEVICES=4 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --pos_weight ${INIT_POSWEIGHT}

CUDA_VISIBLE_DEVICES=7 python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}