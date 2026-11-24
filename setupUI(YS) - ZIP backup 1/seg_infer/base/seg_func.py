"""
语义分割后处理模块
适配 PP-LiteSeg / PPSeg 模型的 RKNN 输出格式

功能：
1. 将NPU输出的特征图还原为像素级类别标签
2. 可视化分割结果（不同类别用不同颜色）
3. 提取赛道区域，用于巡线控制
"""

import cv2
import numpy as np

# ============ 配置区 ============
# 类别颜色 (BGR格式)，可根据你的 label_mapping.txt 修改
# 你的 label_mapping.txt 定义了: 0=background(默认), 1=track(赛道)
SEG_COLORS = {
    0: (0, 0, 0),        # 背景 - 黑色
    1: (0, 255, 0),      # 赛道(track) - 绿色
    2: (0, 0, 255),      # 边线 - 红色
    3: (255, 0, 0),      # 其他 - 蓝色
}

# 输入分辨率（必须与模型训练时的尺寸一致）
# 注意: pp_liteseg.rknn 的输入尺寸是 384x384
# 如果报错 "current input size(786432) > need input size(442368)"
# 说明尺寸不对: 442368 = 384*384*3，需改为 (384, 384)
SEG_INPUT_SIZE = (384, 384)


def seg_post_process(output, orig_shape):
    """
    语义分割后处理：将模型输出转为类别标签图

    参数:
        output: list, RKNN 模型输出列表
        orig_shape: tuple, 原始图像 (height, width)

    返回:
        mask: numpy.ndarray, 每个像素是类别ID (0,1,2...)
    """
    # PP-LiteSeg 的输出通常是一个 [1, num_classes, H, W] 的特征图
    # 也可能是 [1, H, W] 的 argmax 结果
    if isinstance(output, (list, tuple)):
        seg_logit = output[0]
    else:
        seg_logit = output

    # 处理不同维度情况
    if len(seg_logit.shape) == 4:
        # [1, C, H, W] → [H, W]
        seg_logit = seg_logit[0]  # 去掉batch维 → [C, H, W]
        mask = np.argmax(seg_logit, axis=0)  # 沿类别维取argmax → [H, W]
    elif len(seg_logit.shape) == 3:
        # [C, H, W] → [H, W]
        mask = np.argmax(seg_logit, axis=0)
    elif len(seg_logit.shape) == 2:
        # 已经是 [H, W] 格式
        mask = seg_logit
    else:
        print(f"[WARN] 未知的输出维度: {seg_logit.shape}")
        mask = np.zeros(orig_shape[:2], dtype=np.uint8)

    # 将mask缩放回原始图像大小
    mask = cv2.resize(mask.astype(np.uint8), (orig_shape[1], orig_shape[0]),
                      interpolation=cv2.INTER_NEAREST)

    return mask


def seg_visualize(image, mask, alpha=0.5):
    """
    将分割结果叠加到原图上可视化

    参数:
        image: 原始图像 (BGR)
        mask: 类别标签图
        alpha: 叠加透明度 (0~1)

    返回:
        overlay: 叠加后的图像
    """
    # 创建彩色mask
    color_mask = np.zeros_like(image)
    for class_id, color in SEG_COLORS.items():
        color_mask[mask == class_id] = color

    # 叠加
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return overlay


def extract_lane_center(mask, class_id=1, image_bottom_ratio=0.7):
    """
    从分割mask中提取赛道中心线，用于巡线控制

    参数:
        mask: 类别标签图
        class_id: 赛道区域的类别ID（你的label_mapping中 1=track）
        image_bottom_ratio: 从图像底部多大比例区域提取中心线

    返回:
        center_x: 赛道中心点的x坐标（相对于图像宽度，0~1）
        center_y: 赛道中心点的y坐标
        offset: 中心线偏离图像中心的偏移量（正值=偏右，负值=偏左）
    """
    h, w = mask.shape

    # 只取图像底部区域（车前方最近的赛道）
    roi_start = int(h * (1 - image_bottom_ratio))
    roi = mask[roi_start:, :]

    # 找到赛道区域的像素
    lane_pixels = np.where(roi == class_id)

    if len(lane_pixels[0]) == 0:
        # 没检测到赛道，返回中心
        return 0.5, h * image_bottom_ratio * 0.5, 0.0

    # 计算赛道中心
    center_x = np.mean(lane_pixels[1]) / w  # 归一化到 0~1
    center_y = np.mean(lane_pixels[0]) + roi_start

    # 偏移量：正值=赛道偏右，负值=偏左
    offset = center_x - 0.5

    return center_x, center_y, offset


def mySegFunc(rknn_lite, img_orig):
    """
    线程池回调函数：语义分割推理

    参数:
        rknn_lite: RKNNLite 实例
        img_orig: 原始图像 (BGR)

    返回:
        result_img: 带分割可视化结果的图像
        mask: 分割标签图
        lane_offset: 赛道中心偏移量
    """
    orig_shape = img_orig.shape[:2]

    # 预处理：缩放到模型输入尺寸
    img_resized = cv2.resize(img_orig, SEG_INPUT_SIZE)
    img_input = np.expand_dims(img_resized, axis=0)  # [1, H, W, 3]

    # NPU 推理
    outputs = rknn_lite.inference(inputs=[img_input])

    # 后处理
    mask = seg_post_process(outputs, orig_shape)

    # 可视化
    result_img = seg_visualize(img_orig, mask, alpha=0.4)

    # 提取巡线信息
    center_x, center_y, lane_offset = extract_lane_center(mask, class_id=1)

    # 在图像上标注中心线
    cx = int(center_x * orig_shape[1])
    cy = int(center_y)
    cv2.circle(result_img, (cx, cy), 5, (0, 255, 255), -1)
    cv2.line(result_img, (cx, 0), (cx, orig_shape[0]), (0, 255, 255), 1)
    cv2.putText(result_img, f"Lane offset: {lane_offset:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return result_img, mask, lane_offset
