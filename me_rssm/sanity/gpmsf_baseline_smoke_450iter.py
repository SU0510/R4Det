# Paired S0 resource reference: same B0 RSSM, bs=2/rank, N4, accumulation=2,
# 3 ranks, 450 iterations and solver. Only the RCFusion arm differs.
_base_ = ['./gpmsf_smoke_450iter.py']
model = dict(RCFusion=dict(
    _delete_=True, type='ConcatConvFusion', img_channels=256, rad_channels=384,
    out_channels=256, kernel_size=3,
    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
    act_cfg=dict(type='ReLU', inplace=True)))
custom_hooks = [dict(type='KLScaleSchedulerHook', start_epoch=0, end_epoch=10,
                     start_value=0.0, end_value=1.0)]
