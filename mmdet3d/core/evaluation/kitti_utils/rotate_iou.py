#####################
# Based on https://github.com/hongzhenwang/RRPN-revise
# Licensed under The MIT License
#
# Modified: replaced numba.cuda with numba CPU jit.
# All division-by-zero paths are guarded.
#####################
import math

import numba
import numpy as np


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@numba.jit(nopython=True)
def _triangle_area(a_x, a_y, b_x, b_y, c_x, c_y):
    return ((a_x - c_x) * (b_y - c_y) - (a_y - c_y) * (b_x - c_x)) * 0.5


@numba.jit(nopython=True)
def _polygon_area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            _triangle_area(int_pts[0], int_pts[1],
                           int_pts[2 * i + 2], int_pts[2 * i + 3],
                           int_pts[2 * i + 4], int_pts[2 * i + 5]))
    return area_val


@numba.jit(nopython=True)
def _sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center_x = 0.0
        center_y = 0.0
        for i in range(num_of_inter):
            center_x += int_pts[2 * i]
            center_y += int_pts[2 * i + 1]
        center_x /= num_of_inter
        center_y /= num_of_inter
        vs = np.zeros(16, dtype=np.float32)
        for i in range(num_of_inter):
            v_x = int_pts[2 * i] - center_x
            v_y = int_pts[2 * i + 1] - center_y
            d = math.sqrt(v_x * v_x + v_y * v_y)
            if d > 1e-12:
                v_x = v_x / d
                v_y = v_y / d
            if v_y < 0.0:
                v_x = -2.0 - v_x
            vs[i] = v_x
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1
                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@numba.jit(nopython=True)
def _line_segment_intersection(pts1, pts2, i, j, temp_pts):
    """Returns True and fills temp_pts if edge i of pts1 intersects edge j of pts2."""
    a_x = pts1[2 * i]
    a_y = pts1[2 * i + 1]
    b_x = pts1[2 * ((i + 1) % 4)]
    b_y = pts1[2 * ((i + 1) % 4) + 1]
    c_x = pts2[2 * j]
    c_y = pts2[2 * j + 1]
    d_x = pts2[2 * ((j + 1) % 4)]
    d_y = pts2[2 * ((j + 1) % 4) + 1]

    ba_x = b_x - a_x
    ba_y = b_y - a_y
    da_x = d_x - a_x
    ca_x = c_x - a_x
    da_y = d_y - a_y
    ca_y = c_y - a_y

    acd = da_y * ca_x > ca_y * da_x
    bcd = (d_y - b_y) * (c_x - b_x) > (c_y - b_y) * (d_x - b_x)
    if acd != bcd:
        abc = ca_y * ba_x > ba_y * ca_x
        abd = da_y * ba_x > ba_y * da_x
        if abc != abd:
            dc_x = d_x - c_x
            dc_y = d_y - c_y
            abba = a_x * b_y - b_x * a_y
            cddc = c_x * d_y - d_x * c_y
            dh = ba_y * dc_x - ba_x * dc_y
            if dh != 0.0:
                dx = abba * dc_x - ba_x * cddc
                dy = abba * dc_y - ba_y * cddc
                temp_pts[0] = dx / dh
                temp_pts[1] = dy / dh
                return True
    return False


@numba.jit(nopython=True)
def _point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]
    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]
    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]
    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1
    return abab >= abap and abap >= 0.0 and adad >= adap and adap >= 0.0


@numba.jit(nopython=True)
def _quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if _point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if _point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = np.zeros(2, dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if _line_segment_intersection(pts1, pts2, i, j, temp_pts):
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1
    return num_of_inter


@numba.jit(nopython=True)
def _rbbox_to_corners(rbbox):
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners = np.zeros(8, dtype=np.float32)
    c_x0, c_x1, c_x2, c_x3 = -x_d / 2.0, -x_d / 2.0, x_d / 2.0, x_d / 2.0
    c_y0, c_y1, c_y2, c_y3 = -y_d / 2.0, y_d / 2.0, y_d / 2.0, -y_d / 2.0
    corners[0] = a_cos * c_x0 + a_sin * c_y0 + center_x
    corners[1] = -a_sin * c_x0 + a_cos * c_y0 + center_y
    corners[2] = a_cos * c_x1 + a_sin * c_y1 + center_x
    corners[3] = -a_sin * c_x1 + a_cos * c_y1 + center_y
    corners[4] = a_cos * c_x2 + a_sin * c_y2 + center_x
    corners[5] = -a_sin * c_x2 + a_cos * c_y2 + center_y
    corners[6] = a_cos * c_x3 + a_sin * c_y3 + center_x
    corners[7] = -a_sin * c_x3 + a_cos * c_y3 + center_y
    return corners


@numba.jit(nopython=True)
def _inter(rbbox1, rbbox2):
    corners1 = _rbbox_to_corners(rbbox1)
    corners2 = _rbbox_to_corners(rbbox2)
    intersection_corners = np.zeros(16, dtype=np.float32)
    num_inter = _quadrilateral_intersection(corners1, corners2, intersection_corners)
    _sort_vertex_in_convex_polygon(intersection_corners, num_inter)
    return _polygon_area(intersection_corners, num_inter)


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Rotated box IoU — CPU numba implementation.

    Args:
        boxes (np.ndarray): shape (N, 5), format [cx, cy, w, l, angle].
        query_boxes (np.ndarray): shape (K, 5).
        criterion (int): -1=Iou, 0=inter/area1, 1=inter/area2, else=area_inter.
    """
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N, K = boxes.shape[0], query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    for n in range(N):
        area1 = boxes[n, 2] * boxes[n, 3]
        if area1 <= 0.0:
            continue
        for k in range(K):
            area2 = query_boxes[k, 2] * query_boxes[k, 3]
            if area2 <= 0.0:
                continue
            inter = _inter(boxes[n], query_boxes[k])
            if criterion == -1:
                union = area1 + area2 - inter
                iou[n, k] = inter / union if union > 0.0 else 0.0
            elif criterion == 0:
                iou[n, k] = inter / area1
            elif criterion == 1:
                iou[n, k] = inter / area2
            else:
                iou[n, k] = inter
    return iou