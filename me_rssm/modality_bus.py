"""Per-frame modality side-channel bus for the Motion-Evidence RSSM.

Why this exists
---------------
The unmodified ``R4Det.extract_feat`` pipeline wires modules in a fixed
sequence:

    pts_voxel_encoder -> pts_middle_encoder (scatter) -> pts_backbone/neck
    -> RCFusion(img_bev, radar_bev) -> temporal_fusion(fused_bev)

Only the *fused* BEV reaches ``temporal_fusion``. The radar Doppler /
velocity evidence (raw point-level radial velocity) and the two un-fused
modality BEVs are discarded upstream, so a temporal module plugged into the
standard interface can never see them.

Rather than duplicating ~200 lines of ``extract_feat`` in a detector
subclass (fragile) or modifying existing files (forbidden by the research
protocol), the producers stash their extra outputs here and the temporal
module reads them in the same forward call.

Contract (written every ``extract_feat`` call, read exactly once after):

    'velocity_bev': (B, C_motion, ny, nx)  raw scatter of per-pillar velocity
                    statistics at 0.16 m resolution. The scatter canvas is
                    (ny=output_shape[0], nx=output_shape[1]), i.e. rows are
                    the *y* axis (496 = 2*248) and columns the *x* axis
                    (432 = 2*216). Consumers must avg_pool(2) ->
                    (B, C, 248, 216) then permute to (B, C, 216, 248) to
                    match the temporal-fusion canvas (rows = x, cols = y).
    'camera_bev'  : (B, 256, 248, 216) camera BEV before the (y,x)->(x,y)
                    permute applied in ``extract_feat``.
    'radar_bev'   : (B, 384, 248, 216) radar BEV, same convention.

Because ``extract_feat`` runs exactly once per frame (burn-in, BPTT window
and current frame alike) and the temporal fusion is invoked at the very end
of the same call, a reader always observes the values written for the frame
being processed. Nothing here carries gradients that training relies on;
the velocity statistics are computed from raw (no-grad) point inputs.
"""

__all__ = ['ModalityContext']


class ModalityContext(object):
    """Class-level namespace holding the current frame's modality extras."""

    _store = {}

    @classmethod
    def set(cls, key, value):
        cls._store[key] = value

    @classmethod
    def get(cls, key, default=None):
        return cls._store.get(key, default)

    @classmethod
    def clear(cls):
        cls._store.clear()
