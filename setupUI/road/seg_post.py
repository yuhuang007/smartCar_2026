# road/seg_post.py
# 分割后处理：提取 mask、计算中心偏移、可视化叠加

import cv2
import numpy as np

def prob_to_mask(prob_track, thresh=0.5):
    """单通道概率图 -> 二值mask"""
    return (prob_track > thresh).astype(np.uint8)

def lane_offset_from_mask(mask, roi_y_ratio=0.5):
    """
    计算赛道中心偏移量
    roi_y_ratio: 取图像下半部分（越小越偏下方）
    返回：
        offset >0 表示目标在右侧，需要右转
        offset <0 表示目标在左侧，需要左转
    """
    h, w = mask.shape
    roi = mask[int(h * roi_y_ratio):, :]
    ys, xs = np.where(roi > 0)

    if len(xs) == 0:
        return 0.0

    cx = xs.mean()
    offset = cx - (w / 2.0)
    return float(offset)

def overlay_mask(img_bgr, mask, alpha=0.35, color=(0, 255, 0)):
    """在原图上叠加半透明mask（带尺寸保护，避免全屏染色）"""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    if mask is None:
        return vis

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mask_bool = mask.astype(bool)
    pos_ratio = float(mask_bool.sum()) / float(h * w + 1e-6)
    print(f"[VIS] mask positive ratio: {pos_ratio:.4f}")

    # 如果 mask 覆盖率太高（例如>80%），降低 alpha 防止“全绿”
    if pos_ratio > 0.8:
        alpha = 0.15

    color_layer = np.zeros_like(vis, dtype=np.uint8)
    color_layer[:] = color

    blended = cv2.addWeighted(vis[mask_bool], 1 - alpha, color_layer[mask_bool], alpha, 0)
    if blended is None:
        return vis

    vis[mask_bool] = blended
    return vis