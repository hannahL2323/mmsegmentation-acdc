#!/bin/bash
# SegFormer-B0 训练脚本

conda activate openmmlab
cd /home/deng/mmsegmentation
export CUDA_VISIBLE_DEVICES=1

python tools/train.py /home/deng/acdc_project/configs/segformer/segformer_acdc.py --work-dir work_dirs/segformer_acdc --auto-resume

echo "SegFormer-B0 训练完成！"