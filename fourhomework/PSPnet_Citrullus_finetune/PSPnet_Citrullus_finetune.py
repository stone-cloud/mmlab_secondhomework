norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=(512, 512),
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        size=(512, 512),
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255),
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        'E:\\AIsecond\\second_mmseg\\mmsegmentation\\work_dirs\\PSPnet_Citrullus_finetune\\iter_4000.pth'
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'BaseSegDataset'
metainfo = dict(
    classes=('red', 'green', 'white', 'seed-black', 'seed-white'),
    palette=[(220, 20, 60), (110, 50, 60), (220, 110, 20), (20, 60, 60),
             (20, 110, 60)])
data_root = 'E:/AIsecond/second_mmseg/mmsegmentation/data/Watermelon87_Semantic_Seg_Mask/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale_factor': 0.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 0.75,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.0,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.25,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.75,
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 0.0,
                        'direction': 'horizontal'
                    }, {
                        'type': 'RandomFlip',
                        'prob': 1.0,
                        'direction': 'horizontal'
                    }], [{
                        'type': 'LoadAnnotations'
                    }], [{
                        'type': 'PackSegInputs'
                    }]])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=
        'E:/AIsecond/second_mmseg/mmsegmentation/data/Watermelon87_Semantic_Seg_Mask/',
        metainfo=dict(
            classes=('red', 'green', 'white', 'seed-black', 'seed-white'),
            palette=[(220, 20, 60), (110, 50, 60), (220, 110, 20),
                     (20, 60, 60), (20, 110, 60)]),
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomResize',
                scale=(2048, 1024),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        metainfo=dict(
            classes=('red', 'green', 'white', 'seed-black', 'seed-white'),
            palette=[(220, 20, 60), (110, 50, 60), (220, 110, 20),
                     (20, 60, 60), (20, 110, 60)]),
        reduce_zero_label=True,
        data_root=
        'E:/AIsecond/second_mmseg/mmsegmentation/data/Watermelon87_Semantic_Seg_Mask/',
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        metainfo=dict(
            classes=('red', 'green', 'white', 'seed-black', 'seed-white'),
            palette=[(220, 20, 60), (110, 50, 60), (220, 110, 20),
                     (20, 60, 60), (20, 110, 60)]),
        reduce_zero_label=True,
        data_root=
        'E:/AIsecond/second_mmseg/mmsegmentation/data/Watermelon87_Semantic_Seg_Mask/',
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
    save_dir='result')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = 'work_dirs/PSPnet_Citrullus_finetune/best_aAcc_iter_17600.pth'
resume = False
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=400)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=400,
        max_keep_ckpts=5,
        save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True))
launcher = 'none'
work_dir = './work_dirs\\PSPnet_Citrullus_finetune'
