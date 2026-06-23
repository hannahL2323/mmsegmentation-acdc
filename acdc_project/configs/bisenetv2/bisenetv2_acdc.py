_base_ = [
    '/home/deng/mmsegmentation/configs/_base_/models/bisenetv2.py',
    '/home/deng/mmsegmentation/configs/_base_/datasets/cityscapes_1024x1024.py',
    '/home/deng/mmsegmentation/configs/_base_/default_runtime.py',
    '/home/deng/mmsegmentation/configs/_base_/schedules/schedule_160k.py'
]

crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=160000,
        by_epoch=False,
    )
]

optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# ========== ACDC Dataset Configuration ==========
data_root = '/home/deng/datasets/acdc_mmseg'

# Override dataset config from base to point to ACDC
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='CityscapesDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type='CityscapesDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val'
        )
    )
)

test_dataloader = val_dataloader

# ========== Logging and Checkpoints ==========
# Save checkpoint every 8000 iterations (20 checkpoints for 160k)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=8000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))