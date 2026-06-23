
_base_ = 'acdc_project/configs/pidnet/pidnet-s_acdc.py'

# 保留 4 个损失函数，但只让第一个生效（权重设为 1，其他设为 0）
model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0),  # 占位
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0),  # 占位
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0)   # 占位
        ]
    )
)

# 完全禁用验证
val_dataloader = None
val_cfg = None
val_evaluator = None

# 只跑 5 次迭代
train_cfg = dict(type='IterBasedTrainLoop', max_iters=5, val_interval=999999)
