_base_ = './TJ4D-R4Det_ped_highres_centerhead_3x2x2_3e_raw.py'

model = dict(
    ped_center_head=dict(
        test_cfg=dict(
            size_prior_alpha=[
                0.655454, 0.627535, 0.75,
            ])))
