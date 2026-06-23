#!/bin/bash
# PIDNet-S 训练脚本

conda activate openmmlab
cd /home/deng/mmsegmentation
export CUDA_VISIBLE_DEVICES=1

python tools/train.py /home/deng/acdc_project/configs/pidnet/pidnet-s_acdc.py --work-dir work_dirs/pidnet_acdc --auto-resume

echo "PIDNet-S 训练完成！"