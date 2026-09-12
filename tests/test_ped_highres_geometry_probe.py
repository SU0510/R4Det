import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / (
    "ped_highres_geometry_probe.py")
SPEC = importlib.util.spec_from_file_location("ped_probe", MODULE_PATH)
PED_PROBE = importlib.util.module_from_spec(SPEC)
_cuda_is_available = torch.cuda.is_available
try:
    torch.cuda.is_available = lambda: True
    SPEC.loader.exec_module(PED_PROBE)
finally:
    torch.cuda.is_available = _cuda_is_available


def test_index_xy_uses_per_level_voxel_size():
    point = torch.tensor([1.0, -3.0])
    pc_range = [0.0, -39.68, -4.0, 69.12, 39.68, 2.0]

    low = PED_PROBE._index_xy(point, pc_range, [0.32, 0.32])
    high = PED_PROBE._index_xy(point, pc_range, [0.16, 0.16])

    assert tuple(low.tolist()) == (3, 114)
    assert tuple(high.tolist()) == (6, 229)


def test_unwrap_handles_data_container_cycle():
    class CyclicData:
        pass

    container = CyclicData()
    container.data = container

    assert PED_PROBE._unwrap(container) is container


def test_unpack_sample_accepts_collated_sequence_dims():
    class Boxes:
        def __init__(self):
            self.tensor = torch.zeros(7, 7)

    sample = {
        "points": [torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3)],
        "img": torch.zeros(1, 3, 3, 8, 8),
        "img_metas": [[{"is_prev_frame_valid": False},
                       {"is_prev_frame_valid": True},
                       {"is_prev_frame_valid": True}]],
        "gt_bboxes_3d": [Boxes(), Boxes(), Boxes()],
        "gt_labels_3d": [torch.zeros(0, dtype=torch.long)] * 3,
    }

    points, img, metas, boxes, labels = PED_PROBE._unpack_sample(sample)

    assert len(points) == 3
    assert img.shape == (3, 3, 8, 8)
    assert len(metas) == 3
    assert len(boxes) == 3
    assert len(labels) == 3


def test_select_ped_indices_uses_annotations():
    class Dataset:
        data_infos = [
            {"annos": {"name": ["Car"]}},
            {"annos": {"name": ["Pedestrian", "Car"]}},
            {"annos": {"name": ["Truck"]}},
            {"annos": {"name": ["Pedestrian"]}},
        ]

    indices = PED_PROBE._select_ped_indices(Dataset(), limit=1)

    assert indices == [1]


def test_crop_patch_rejects_out_of_bounds_and_returns_aligned_shapes():
    boxes = torch.tensor([[1.0, -3.0, 0.0, 0.6, 0.8, 1.7, 0.2]])
    highres = torch.arange(1 * 2 * 432 * 496, dtype=torch.float32).view(
        1, 2, 432, 496)
    radar = torch.arange(1 * 3 * 432 * 496, dtype=torch.float32).view(
        1, 3, 432, 496)
    pc_range = [0.0, -39.68, -4.0, 69.12, 39.68, 2.0]

    patch = PED_PROBE._crop_patch(
        highres, radar, boxes[0], [0.16, 0.16], pc_range, 9)

    assert patch is not None
    assert patch[0].shape == (2, 9, 9)
    assert patch[1].shape == (3, 9, 9)
    assert PED_PROBE._crop_patch(
        highres, radar,
        torch.tensor([-1.0, 0.0, 0.0, 0.6, 0.8, 1.7, 0.2]),
        [0.16, 0.16], pc_range, 9) is None


def test_geometry_head_shape_and_gradient():
    head = PED_PROBE.GeometryHead(input_channels=5, hidden_channels=8)
    inputs = torch.randn(3, 5, 9, 9)

    outputs = head(inputs)
    outputs.square().mean().backward()

    assert outputs.shape == (3, 4)
    assert all(parameter.grad is not None for parameter in head.parameters())


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
