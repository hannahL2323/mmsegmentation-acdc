_base_ = [
    '/home/deng/mmsegmentation/configs/_base_/models/segformer_mit-b0.py',
    '/home/deng/mmsegmentation/configs/_base_/datasets/cityscapes_1024x1024.py',
    '/home/deng/mmsegmentation/configs/_base_/default_runtime.py',
    '/home/deng/mmsegmentation/configs/_base_/schedules/schedule_160k.py'
]

crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

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