import argparse
import os
import torch
from mmcv.parallel import collate
from mmcv import Config
from mmdet3d.models import build_detector
from mmdet3d.datasets import build_dataset


def topk_xy(heat, k=10):
    batch, cat, h, w = heat.shape
    values, inds = torch.topk(heat.view(batch, -1), k, dim=-1)
    ys = inds // w
    xs = inds % w
    return list(zip(values[0].tolist(), ys[0].tolist(), xs[0].tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--num-samples', type=int, default=1)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    project = os.path.splitext(os.path.basename(args.config))[0]
    figures_path = os.path.join('work_dirs', project, 'figures_path')
    os.makedirs(figures_path, exist_ok=True)
    cfg.model.update(meta_info={
        'figures_path': figures_path,
        'project_name': project,
    })
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    dataset = build_dataset(cfg.data.test)
    for sample_idx in range(args.num_samples):
        data = dataset[sample_idx]
        collated = collate([data], samples_per_gpu=1)
        with torch.no_grad():
            def unwrap_dc(value):
                if hasattr(value, 'data') and not isinstance(value, (list, tuple)):
                    return value.data
                return value

            def to_cuda(value):
                return value.cuda() if torch.is_tensor(value) else value

            img = unwrap_dc(collated['img'])[0].cuda()
            if img.dim() == 5:
                img = img[:, -1]
            points = to_cuda(unwrap_dc(collated['points']))
            points = [points[-1][-1][-1].cuda()]
            img_metas = unwrap_dc(collated['img_metas'])
            img_metas = [img_metas[-1][-1][-1]]
            feature_dict = model.extract_feat(
                points, img, img_metas, feat_or_dict=1)
            fused = feature_dict['pts_feats'][0]
            stage1_out = model.ped_center_head([fused])[0][0]
            highres = model._make_ped_highres_feats([fused])[0]
            highres_out = model.ped_center_head([highres])[0][0]

        print('sample', sample_idx)
        print('fused', tuple(fused.shape), 'highres', tuple(highres.shape))
        print('lowres_heatmap', tuple(stage1_out['heatmap'].shape))
        print('highres_heatmap', tuple(highres_out['heatmap'].shape))
        print('lowres_top10_xy', topk_xy(stage1_out['heatmap'].sigmoid()))
        print('highres_top10_xy', topk_xy(highres_out['heatmap'].sigmoid()))


if __name__ == '__main__':
    main()
