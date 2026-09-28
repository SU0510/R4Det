
import numpy as np
import torch


def gaussian_radius_batch(height, width, min_overlap=0.5):
    """Vectorised :func:`gaussian_radius` for tensors of box sizes.

    ``gaussian_radius`` ends in a Python ``min()`` over three tensors and
    therefore cannot accept a batch. This is the same closed form with
    ``torch.minimum`` instead, kept next to the scalar version so both stay
    in sync (the unit test asserts element-wise equality).
    """
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + (b1.square() - 4 * a1 * c1).clamp_min(0).sqrt()) / 2
    a2, b2 = 4.0, 2.0 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + (b2.square() - 4 * a2 * c2).clamp_min(0).sqrt()) / 2
    a3, b3 = 4.0 * min_overlap, -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + (b3.square() - 4 * a3 * c3).clamp_min(0).sqrt()) / 2
    return torch.minimum(torch.minimum(r1, r2), r3)


def bev_centers_to_heatmap(gt_bboxes_3d,
                           gt_labels_3d,
                           num_classes,
                           out_size,
                           point_cloud_range,
                           min_radius=2,
                           device=None,
                           dtype=torch.float32):
    """Rasterize 3D box centers into a small multi-class Gaussian heatmap.

    Used by the route-C object-centric latent supervision: the auxiliary head
    predicts this map from the low-dimensional posterior correction so the
    latent is asked for detection-relevant content instead of a batch-common
    BEV template.

    The output grid follows the latent/feature convention used everywhere in
    this detector: ``H`` indexes the ``x`` axis and ``W`` indexes the ``y``
    axis of the ego BEV. Each cell therefore spans
    ``(x_max - x_min) / H`` by ``(y_max - y_min) / W``.

    Args:
        gt_bboxes_3d (list): Per-sample 3D boxes. Any object exposing a
            ``gravity_center`` ``(N, 3)`` tensor works.
        gt_labels_3d (list[torch.Tensor]): Per-sample integer class labels.
        num_classes (int): Number of heatmap channels.
        out_size (tuple[int, int]): ``(H, W)`` of the output heatmap.
        point_cloud_range (list[float]): ``[x_min, y_min, z_min, x_max, ...]``.
        min_radius (int): Minimum Gaussian radius in heatmap cells.
        device (torch.device, optional): Output device. Defaults to the label
            device of the first sample, or CPU.
        dtype (torch.dtype): Output dtype.

    Returns:
        torch.Tensor: ``(B, num_classes, H, W)`` heatmap in ``[0, 1]``.
    """
    out_h, out_w = int(out_size[0]), int(out_size[1])
    if len(gt_bboxes_3d) == 0:
        raise ValueError('bev_centers_to_heatmap needs at least one sample')
    if device is None:
        device = gt_labels_3d[0].device

    heatmap = torch.zeros(
        (len(gt_bboxes_3d), num_classes, out_h, out_w),
        device=device, dtype=dtype)

    x_min, y_min = float(point_cloud_range[0]), float(point_cloud_range[1])
    x_max, y_max = float(point_cloud_range[3]), float(point_cloud_range[4])
    stride_x = (x_max - x_min) / float(out_h)
    stride_y = (y_max - y_min) / float(out_w)

    # All box sizes go to the host once, then radii are computed with one
    # vectorised call to the same `gaussian_radius` formula the detector's
    # CenterHead uses. A per-box call on CUDA tensors would synchronise once
    # per object per training iteration.
    all_dims = []
    for boxes in gt_bboxes_3d:
        if boxes.gravity_center.numel() == 0:
            all_dims.append(torch.zeros(0, 2))
        else:
            dims = boxes.tensor[:, 3:6].detach().float().cpu()
            all_dims.append(torch.stack(
                [dims[:, 0] / stride_x, dims[:, 1] / stride_y], dim=1))
    if any(size.numel() for size in all_dims):
        sizes = torch.cat(all_dims, dim=0)
        all_radii = gaussian_radius_batch(
            sizes[:, 0], sizes[:, 1],
            min_overlap=0.5).clamp_min(float(min_radius)).int()
    else:
        all_radii = torch.zeros(0, dtype=torch.int32)
    radius_cursor = 0

    # Radius -> kernel cache, so a handful of distinct radii never rebuild
    # their Gaussian on every object.
    kernel_cache = {}

    def kernel_for(radius):
        if radius not in kernel_cache:
            diameter = 2 * radius + 1
            sigma = max(float(diameter) / 6.0, 1e-3)
            offsets = torch.arange(
                -radius, radius + 1, device=device, dtype=dtype)
            dx = offsets.view(-1, 1).square()
            dy = offsets.view(1, -1).square()
            kernel_cache[radius] = torch.exp(
                -(dx + dy) / (2.0 * sigma * sigma))
        return kernel_cache[radius]

    for b, (boxes, labels) in enumerate(zip(gt_bboxes_3d, gt_labels_3d)):
        centers = boxes.gravity_center
        if centers.numel() == 0:
            continue
        # Pull GT to host once per sample: every box only needs Python
        # scalars for its integer cell and radius, and looping on CUDA
        # tensors would otherwise force one synchronisation per box.
        centers_cpu = centers.detach().float().cpu().tolist()
        dims_cpu = boxes.tensor[:, 3:6].detach().float().cpu().tolist()
        labels_cpu = labels.detach().cpu().tolist()
        num_boxes = len(centers_cpu)
        radii = all_radii[radius_cursor:radius_cursor + num_boxes]
        radius_cursor += num_boxes
        for k, (cx, cy, _cz) in enumerate(centers_cpu):
            cls_id = int(labels_cpu[k])
            if cls_id < 0 or cls_id >= num_classes:
                continue
            row_i = int((cx - x_min) / stride_x)
            col_i = int((cy - y_min) / stride_y)
            if not (0 <= row_i < out_h and 0 <= col_i < out_w):
                continue
            size_x = dims_cpu[k][0] / stride_x
            size_y = dims_cpu[k][1] / stride_y
            if size_x <= 0 or size_y <= 0:
                continue
            radius = max(int(min_radius), int(radii[k]))
            kernel = kernel_for(radius)
            row_lo, row_hi = max(0, row_i - radius), min(
                out_h, row_i + radius + 1)
            col_lo, col_hi = max(0, col_i - radius), min(
                out_w, col_i + radius + 1)
            k_row_lo = row_lo - (row_i - radius)
            k_col_lo = col_lo - (col_i - radius)
            patch = heatmap[
                b, cls_id, row_lo:row_hi, col_lo:col_hi]
            kernel_patch = kernel[
                k_row_lo:k_row_lo + (row_hi - row_lo),
                k_col_lo:k_col_lo + (col_hi - col_lo)]
            torch.maximum(patch, kernel_patch, out=patch)

    return heatmap


def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float, optional): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    # print(shape,m,n)
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_heatmap_gaussian_feat(heatmap, center, radius, feat, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    # gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[-2:]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    heatmap[:, y - top:y + bottom, x - left:x + right] = feat.view(-1, 1, 1).expand_as(heatmap[:, y - top:y + bottom, x - left:x + right])

    return heatmap

# def draw_heatmap_gaussian_feat(heatmap, center, radius, k=1):
#     """Get gaussian masked heatmap.

#     Args:
#         heatmap (torch.Tensor): Heatmap to be masked.
#         center (torch.Tensor): Center coord of the heatmap.
#         radius (int): Radius of gaussian.
#         K (int, optional): Multiple of masked_gaussian. Defaults to 1.

#     Returns:
#         torch.Tensor: Masked heatmap.
#     """
#     diameter = 2 * radius + 1
#     gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

#     x, y = int(center[0]), int(center[1])

#     height, width = heatmap.shape[-2:]

#     left, right = min(x, radius), min(width - x, radius + 1)
#     top, bottom = min(y, radius), min(height - y, radius + 1)

#     return [left, right, top, bottom]


def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float, optional): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def get_ellip_gaussian_2D(heatmap, center, radius_x, radius_y, k=1):
    """Generate 2D ellipse gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius_x (int): X-axis radius of gaussian kernel.
        radius_y (int): Y-axis radius of gaussian kernel.
        k (int, optional): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter_x, diameter_y = 2 * radius_x + 1, 2 * radius_y + 1
    gaussian_kernel = ellip_gaussian2D((radius_x, radius_y),
                                       sigma_x=diameter_x / 6,
                                       sigma_y=diameter_y / 6,
                                       dtype=heatmap.dtype,
                                       device=heatmap.device)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius_y - top:radius_y + bottom,
                                      radius_x - left:radius_x + right]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap


def ellip_gaussian2D(radius,
                     sigma_x,
                     sigma_y,
                     dtype=torch.float32,
                     device='cpu'):
    """Generate 2D ellipse gaussian kernel.

    Args:
        radius (tuple(int)): Ellipse radius (radius_x, radius_y) of gaussian
            kernel.
        sigma_x (int): X-axis sigma of gaussian function.
        sigma_y (int): Y-axis sigma of gaussian function.
        dtype (torch.dtype, optional): Dtype of gaussian tensor.
            Default: torch.float32.
        device (str, optional): Device of gaussian tensor.
            Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius_y + 1) * (2 * radius_x + 1)`` shape.
    """
    x = torch.arange(
        -radius[0], radius[0] + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius[1], radius[1] + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x) / (2 * sigma_x * sigma_x) - (y * y) /
         (2 * sigma_y * sigma_y)).exp()
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0

    return h
