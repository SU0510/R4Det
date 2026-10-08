"""One-shot calibration of the DC-RSSM-1 physical evidence field constants.

docs/13 L3: alpha1/beta1 (c_mot0) and alpha2/beta2 (dyn0) are FIXED
calibration constants -- calibrated once from val quantiles, never
trained, no hyperparameter sweep. This script performs that one-shot
calibration.

Method (quantile-based, parameter-free):
  1. Run the parameter-free producer chain (voxelize ->
     RadarPillarFeatureNetMotion -> PointPillarsScatterMotion) on N val
     frames; read the 'velocity_bev' canvas (channels: 4 v_std, 5 SNR,
     6 normalized occupancy/count).
  2. s = count * SNR and d = v_std * count over MEASURED cells
     (count > 0) -- the inputs of the two sigmoids.
  3. alpha = 1 / (q75 - q25) of the measured population: the interquartile
     range maps onto the sigmoid's central band, so the transition width
     is set by the data itself.
     beta = alpha * q50: the knee sits at the median measured cell, so
     half the measured cells land above 0.5 and empty cells (s = d = 0)
     map to sigma(-beta), which is low whenever the measured IQR is small
     relative to the median (self-tuning).
  4. Print the constants + operating-point diagnostics (fraction of
     measured cells above 0.5, where empty cells land), save JSON, and
     optionally write the constants into the DC-RSSM-1 config.

Usage (read-only unless --write-config):
  CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/calibrate_evidence_field.py \
    --config me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_dcrssm1.py \
    --num-frames 300 --output me_rssm/sanity/evidence_field_calibration.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

_CH_V_STD = 4
_CH_SNR = 5
_CH_COUNT = 6


def parse_args():
    p = argparse.ArgumentParser(
        description='One-shot calibration of the L3 evidence field.')
    p.add_argument('--config', required=True)
    p.add_argument('--gpu-id', type=int, default=5)
    p.add_argument('--num-frames', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', required=True)
    p.add_argument('--write-config', action='store_true',
                   help='write the calibrated constants into the config '
                        '(default: print/save only)')
    return p.parse_args()


def main():
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_detector

    from me_rssm.modality_bus import ModalityContext

    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': '/tmp/r4det_ev_calibration_figures',
        'project_name': 'TJ4D-evidence-field-calibration',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    model = model.cuda().eval()

    n = min(args.num_frames, len(dataset))
    stride = max(1, len(dataset) // n)
    indices = list(range(0, len(dataset), stride))[:n]

    s_vals, d_vals, count_means, snr_means = [], [], [], []
    n_empty_cells = 0
    n_cells = 0
    with torch.no_grad():
        for i, index in enumerate(indices):
            sample = dataset[index]
            points = sample['points']
            img_metas = sample['img_metas']
            if hasattr(points, 'data'):
                points = points.data
            if hasattr(img_metas, 'data'):
                img_metas = img_metas.data
            # producer chain only (voxelize -> encoders -> scatter); the
            # backbone/neck/image branch are unnecessary for the canvas
            pts = [points[0].cuda()]
            voxels, num_points, coors = model.voxelize(pts)
            voxel_features = model.pts_voxel_encoder(
                voxels, num_points, coors)
            model.pts_middle_encoder(voxel_features, coors, 1)
            canvas = ModalityContext.get('velocity_bev')
            ModalityContext.clear()
            if canvas is None:
                continue
            count = canvas[:, _CH_COUNT].flatten()
            snr = canvas[:, _CH_SNR].flatten()
            v_std = canvas[:, _CH_V_STD].flatten()
            measured = count > 0
            n_cells += count.numel()
            n_empty_cells += int((~measured).sum().item())
            if measured.any():
                s_vals.append((count[measured] * snr[measured]).cpu())
                d_vals.append((count[measured] * v_std[measured]).cpu())
                count_means.append(count[measured].mean().item())
                snr_means.append(snr[measured].mean().item())
            if (i + 1) % 100 == 0:
                print(f'[{i + 1}/{len(indices)}] frames done')

    s = torch.cat(s_vals).numpy()
    d = torch.cat(d_vals).numpy()

    def calibrate(x):
        q25, q50, q75 = np.percentile(x, [25, 50, 75])
        iqr = max(q75 - q25, 1e-6)
        alpha = 1.0 / iqr
        beta = alpha * q50
        return alpha, beta, (q25, q50, q75)

    def calibrate_sparse_positive(x):
        """Calibration for a sparse positive population (the dyn field).

        v_std * count is exactly 0 for ~80% of measured cells (static
        radar) and positive for ~20% (dynamic cells) -- measured on 300
        val frames (d>0 frac 0.196). The full-population IQR is 0, so
        the dense calibrate() degenerates (alpha -> inf). Calibrate on
        the POSITIVE-ONLY quantiles instead: the sigmoid knee lands at
        the median DYNAMIC cell; static cells (d=0) map to
        sigma(-beta2) ~ 0.32, giving the field a bimodal structure.
        """
        pos = x[x > 1e-6]
        if pos.size < 100:
            raise ValueError(
                'too few positive dyn cells to calibrate '
                f'({pos.size}); the dynamic evidence would be degenerate')
        q25, q50, q75 = np.percentile(pos, [25, 50, 75])
        iqr = max(q75 - q25, 1e-9)
        alpha = 1.0 / iqr
        beta = alpha * q50
        return alpha, beta, (q25, q50, q75), pos.size

    a1, b1, sq = calibrate(s)
    a2, b2, dq, n_dyn = calibrate_sparse_positive(d)

    frac_s_above = float((a1 * s - b1 > 0).mean())
    frac_d_above = float((a2 * d - b2 > 0).mean())
    empty_c_mot = float(1.0 / (1.0 + np.exp(b1)))
    static_dyn = float(1.0 / (1.0 + np.exp(b2)))

    result = {
        'config': args.config,
        'num_frames': len(indices),
        'cells_total': n_cells,
        'cells_empty_frac': n_empty_cells / max(n_cells, 1),
        's_quantiles_q25_q50_q75': [float(v) for v in sq],
        'd_positive_quantiles_q25_q50_q75': [float(v) for v in dq],
        'd_positive_frac': float((d > 1e-6).mean()),
        'c_mot_alpha1': float(a1),
        'c_mot_beta1': float(b1),
        'dyn_alpha2': float(a2),
        'dyn_beta2': float(b2),
        'operating_point': {
            'measured_frac_above_knee_c_mot': frac_s_above,
            'measured_frac_above_knee_dyn': frac_d_above,
            'empty_cell_c_mot': empty_c_mot,
            'static_cell_dyn': static_dyn,
            'dyn_cells_used_for_calibration': int(n_dyn),
        },
    }
    print(json.dumps(result, indent=2))
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print('[json]', args.output)

    if args.write_config:
        with open(args.config) as f:
            text = f.read()
        for key, value in (('c_mot_alpha1', a1), ('c_mot_beta1', b1),
                           ('dyn_alpha2', a2), ('dyn_beta2', b2)):
            text = _replace_constant(text, key, value)
        with open(args.config, 'w') as f:
            f.write(text)
        print('[config] constants written into', args.config)


def _replace_constant(text, key, value):
    """Replace '<key>=<float>' in the config (one occurrence)."""
    import re
    pattern = rf'(\b{key}=\s*)[0-9.eE+-]+'
    new_text, count = re.subn(pattern, rf'\g<1>{value:.6f}', text)
    if count != 1:
        raise ValueError(
            f'expected exactly one occurrence of {key}=, found {count}')
    return new_text


if __name__ == '__main__':
    main()
