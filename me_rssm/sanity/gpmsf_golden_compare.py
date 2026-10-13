"""Save/compare a golden reference of the GP-MSF toy forward+backward.

Used to prove a *mechanism-preserving* refactor: run ``save`` before the edit
and ``compare`` after it, on the same CPU toy contract.  The head-vectorised
rewrite (training record section 102) reproduced the pre-refactor M_R/M_I and
output bit-for-bit with gradients within 9e-8.

    python me_rssm/sanity/gpmsf_golden_compare.py save /tmp/ref.pt
    python me_rssm/sanity/gpmsf_golden_compare.py compare /tmp/ref.pt
"""
import sys
import torch
sys.path.insert(0, '/home/lurui/workspace/R4Det')
import me_rssm                             # noqa: E402,F401
import mmdet3d.models                      # noqa: E402,F401
from me_rssm.sanity import gpmsf_cpu_checks as toy   # noqa: E402


def run():
    torch.manual_seed(0)
    img, rad = toy.bev_inputs()
    geo = toy.make_geo()
    out = {}
    for tag, chunk, ckpt, stats in (('full', 0, False, True), ('c5', 5, True, True)):
        fz = toy.build(nonzero_out=True, query_chunk=chunk,
                       grad_checkpoint=ckpt, collect_stats=stats)
        fz.train()
        m_r, m_i, o = toy._capture_messages(fz, img, rad, geo, grad=True)
        (o * torch.arange(o.numel(), dtype=o.dtype).view_as(o) / o.numel()).sum().backward()
        out[tag] = dict(
            m_r=m_r.clone(), m_i=m_i.clone(), o=o.detach().clone(),
            stats={k: v for k, v in (fz.last_stats or {}).items()},
            grads={n: p.grad.detach().clone()
                   for n, p in fz.named_parameters() if p.grad is not None},
            keys=[n for n, _ in fz.named_parameters()])
    return out


if __name__ == '__main__':
    mode, path = sys.argv[1], sys.argv[2]
    data = run()
    if mode == 'save':
        torch.save(data, path)
        print('saved', path)
    else:
        ref = torch.load(path)
        worst = 0.0
        for tag in ('full', 'c5'):
            for field in ('m_r', 'm_i', 'o'):
                d = float((ref[tag][field] - data[tag][field]).abs().max())
                worst = max(worst, d)
                print(f'{tag}.{field}: max|d| = {d:.3e}')
            assert ref[tag]['keys'] == data[tag]['keys'], 'parameter key order changed'
            for name, g in ref[tag]['grads'].items():
                d = float((g - data[tag]['grads'][name]).abs().max())
                worst = max(worst, d)
                if d > 1e-6:
                    print(f'  GRAD {tag}.{name}: {d:.3e}')
            sref, snew = ref[tag]['stats'], data[tag]['stats']
            assert set(sref) == set(snew), (set(sref), set(snew))
            for k in sref:
                if isinstance(sref[k], (int, float)) and abs(sref[k] - snew[k]) > 1e-6:
                    print(f'  STAT {tag}.{k}: {sref[k]} -> {snew[k]}')
                elif isinstance(sref[k], list):
                    worst_s = max([abs(a - c) for a, c in zip(sref[k], snew[k])] or [0.0])
                    if worst_s > 1e-6:
                        print(f'  STAT {tag}.{k}: max|d| {worst_s}')
        print(f'worst |delta| = {worst:.3e}')
        assert worst <= 1e-6, f'tolerance exceeded: {worst}'
        print('GOLDEN MATCH')
