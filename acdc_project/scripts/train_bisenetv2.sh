#!/bin/bash
# BiSeNetV2 训练脚本

# 激活环境
conda activate openmmlab

# 进入 MMSegmentation 目录
cd /home/deng/mmsegmentation

# 设置只使用 GPU 1（因为 GPU 0 已被占用）
export CUDA_VISIBLE_DEVICES=1

# 开始训练
python tools/train.py/home/deng/acdc_project/configs/bisenetv2/bisenetv2_acdc.py --work-dir work_dirs/bisenetv2_acdc --auto-resume

echo "BiSeNetV2 训练完成！"