# -*- coding: utf-8 -*-
"""
实时定位脚本

功能：
  1. 加载标定好的单应性矩阵 (homography.npz)
  2. 实时检测智能车上的 ArUco 标记
  3. 计算智能车在地图中的位置 (x, z) 和朝向角度
  4. 在画面上同时显示摄像头画面和俯视地图

使用方法：
  python track.py

前提条件：
  先运行 generate_markers.py 生成标记并摆放好
  再运行 map_calibrate.py 完成标定
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


# ==================== 位置平滑滤波器 ====================
class SmoothingFilter:
    """指数移动平均，减少定位抖动"""

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = np.array(new_value, dtype=np.float64)
        else:
            self.value = self.alpha * np.array(new_value) + (1 - self.alpha) * self.value
        return self.value.copy()

    def reset(self):
        self.value = None


# ==================== 检测器 ====================
def create_detector():
    """创建 ArUco 检测器"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.03
    params.maxMarkerPerimeterRate = 4.0
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    return detector


def detect_markers(detector, frame):
    """检测 ArUco 标记"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    return corners, ids


# ==================== 世界坐标计算 ====================
def pixel_to_world(H, px, py):
    """将像素坐标转换为世界坐标 (x, z)"""
    pt = np.array([[[px, py]]], dtype=np.float32)
    world_pt = cv2.perspectiveTransform(pt, H)
    return world_pt[0, 0, 0], world_pt[0, 0, 1]


def compute_heading(corners_of_marker):
    """
    从标记角点计算朝向角度

    ArUco 角点顺序: [左上, 右上, 右下, 左下]
    标记的"前方"定义为：从底边中点指向顶边中点的方向

    返回: 角度（度），0 = +Z 方向，90 = +X 方向
    """
    pts = corners_of_marker.reshape(-1, 2)
    top_mid = (pts[0] + pts[1]) / 2.0     # 顶边中点
    bottom_mid = (pts[2] + pts[3]) / 2.0  # 底边中点
    direction = top_mid - bottom_mid       # 在图像中的方向

    # 图像坐标系 -> 世界坐标系转换
    # 图像 Y 轴向下，世界 Z 轴向上，需要翻转 Y
    world_dx = direction[0]
    world_dz = -direction[1]

    heading_rad = np.arctan2(world_dx, world_dz)
    heading_deg = np.degrees(heading_rad)
    return heading_deg


# ==================== 俯视地图绘制 ====================
def draw_topdown_map(car_x, car_z, heading_deg, trail, map_w_px=400, map_h_px=500):
    """
    绘制俯视地图

    参数:
      car_x, car_z - 车的世界坐标（米），None 表示未检测到
      heading_deg  - 车的朝向角度
      trail        - 历史轨迹 [(x, z), ...]
    """
    map_img = np.ones((map_h_px, map_w_px, 3), dtype=np.uint8) * 235

    # 像素/米 的比例
    sx = map_w_px / MAP_WIDTH
    sy = map_h_px / MAP_HEIGHT

    # --- 绘制网格 ---
    step = 0.5  # 每 0.5 米一条线
    for xm in np.arange(0, MAP_WIDTH + step, step):
        xpx = int(xm * sx)
        cv2.line(map_img, (xpx, 0), (xpx, map_h_px - 1), (210, 210, 210), 1)
    for zm in np.arange(0, MAP_HEIGHT + step, step):
        zpx = map_h_px - int(zm * sy)
        cv2.line(map_img, (0, zpx), (map_w_px - 1, zpx), (210, 210, 210), 1)

    # --- 绘制边框 ---
    cv2.rectangle(map_img, (0, 0), (map_w_px - 1, map_h_px - 1), (80, 80, 80), 2)

    # --- 绘制地面参考标记 ---
    for mid, (mx, mz) in GROUND_MARKER_POSITIONS.items():
        px = int(mx * sx)
        py = map_h_px - int(mz * sy)
        cv2.circle(map_img, (px, py), 10, (0, 180, 0), -1)
        cv2.circle(map_img, (px, py), 10, (0, 120, 0), 2)
        cv2.putText(map_img, str(mid), (px + 12, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1)

    # --- 绘制轨迹 ---
    for i in range(1, len(trail)):
        tx, tz = trail[i - 1]
        tx2, tz2 = trail[i]
        p1 = (int(tx * sx), map_h_px - int(tz * sy))
        p2 = (int(tx2 * sx), map_h_px - int(tz2 * sy))
        thickness = max(1, 3 - (len(trail) - i) // 20)
        cv2.line(map_img, p1, p2, (200, 120, 50), thickness)

    # --- 绘制智能车（三角形 + 朝向箭头）---
    if car_x is not None and car_z is not None:
        cx = int(car_x * sx)
        cy = map_h_px - int(car_z * sy)

        # 三角形表示车体
        size = 18
        pts = np.array([
            [0, -size],
            [-size * 0.6, size * 0.4],
            [size * 0.6, size * 0.4],
        ], dtype=np.float32)

        # 根据朝向旋转三角形
        rad = np.radians(heading_deg)
        R = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad),  np.cos(rad)]
        ])
        pts = pts @ R.T
        pts[:, 0] += cx
        pts[:, 1] += cy
        pts = pts.astype(np.int32)

        cv2.fillPoly(map_img, [pts], (0, 0, 220))
        cv2.polylines(map_img, [pts], True, (0, 0, 160), 1)

        # 坐标标签
        label = f"({car_x:.2f}, {car_z:.2f})m  {heading_deg:.0f}deg"
        cv2.putText(map_img, label, (cx + 20, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1)

    # --- 坐标轴标签 ---
    cv2.putText(map_img, "X(m)", (map_w_px // 2 - 15, map_h_px - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    cv2.putText(map_img, "Z", (5, map_h_px // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # --- 图例 ---
    cv2.putText(map_img, "Green=Ref  Red=Car", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    return map_img


# ==================== 摄像头画面叠加信息 ====================
def draw_camera_view(frame, corners, ids, car_x, car_z, heading_deg):
    """在摄像头画面上绘制检测结果和定位信息"""
    vis = frame.copy()

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    # 在画面上显示定位信息
    if car_x is not None:
        info = f"Pos: ({car_x:.2f}, {car_z:.2f})m  Heading: {heading_deg:.1f} deg"
        cv2.putText(vis, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
    else:
        cv2.putText(vis, "Car marker NOT detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)

    # 显示检测到的标记数量
    n = len(ids) if ids is not None else 0
    cv2.putText(vis, f"Markers: {n}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return vis


# ==================== 主函数 ====================
def main():
    # --- 检查标定文件 ---
    if not os.path.exists(HOMOGRAPHY_FILE):
        print(f"[错误] 找不到标定文件 '{HOMOGRAPHY_FILE}'！")
        print("  请先运行 map_calibrate.py 完成标定。")
        sys.exit(1)

    # --- 加载单应性矩阵 ---
    data = np.load(HOMOGRAPHY_FILE)
    H = data["H"]
    print(f"已加载标定文件: {HOMOGRAPHY_FILE}")

    # --- 加载摄像头内参（可选）---
    camera_matrix = None
    dist_coeffs = None
    if os.path.exists("camera_calibrate.npz"):
        cam_data = np.load("camera_calibrate.npz")
        camera_matrix = cam_data["camera_matrix"]
        dist_coeffs = cam_data["dist_coeffs"]
        print("已加载摄像头内参，将自动校正镜头畸变")

    # --- 初始化 ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头！")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    detector = create_detector()
    pos_filter = SmoothingFilter(alpha=0.4)
    heading_filter = SmoothingFilter(alpha=0.4)

    trail = []           # 轨迹历史
    max_trail = 300      # 最多保存 300 个轨迹点
    fps_time = time.time()
    fps = 0.0

    print("=" * 55)
    print("  实时定位已启动")
    print("  按 Q 退出  |  按 R 清除轨迹  |  按 S 保存截图")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[错误] 无法读取摄像头画面！")
            break

        # FPS 计算
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - fps_time, 1e-6)
        fps_time = now

        # 镜头畸变校正
        if camera_matrix is not None:
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # 检测标记
        corners, ids = detect_markers(detector, frame)

        car_x, car_z, heading_deg = None, None, 0.0

        # 查找智能车标记
        if ids is not None:
            id_list = ids.flatten().tolist()
            if CAR_MARKER_ID in id_list:
                idx = id_list.index(CAR_MARKER_ID)
                marker_corners = corners[idx]

                # 计算中心像素坐标
                center_px = marker_corners.reshape(-1, 2).mean(axis=0)
                # 转换为世界坐标
                raw_x, raw_z = pixel_to_world(H, center_px[0], center_px[1])

                # 计算朝向
                raw_heading = compute_heading(marker_corners)

                # 平滑滤波
                smoothed = pos_filter.update([raw_x, raw_z])
                car_x, car_z = smoothed[0], smoothed[1]
                heading_deg = heading_filter.update([raw_heading])[0]

                # 记录轨迹
                trail.append((car_x, car_z))
                if len(trail) > max_trail:
                    trail = trail[-max_trail:]
            else:
                pos_filter.reset()
                heading_filter.reset()

        # 绘制摄像头画面
        cam_view = draw_camera_view(frame, corners, ids, car_x, car_z, heading_deg)

        # 绘制俯视地图
        map_view = draw_topdown_map(car_x, car_z, heading_deg, trail)

        # 在摄像头画面上显示 FPS
        cv2.putText(cam_view, f"FPS: {fps:.0f}", (10, cam_view.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # 拼接两个画面（左右并排）
        h = cam_view.shape[0]
        map_resized = cv2.resize(map_view, (int(map_view.shape[1] * h / map_view.shape[0]), h))
        combined = np.hstack([cam_view, map_resized])

        cv2.imshow("ArUco Smart Car Localization", combined)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            trail.clear()
            pos_filter.reset()
            heading_filter.reset()
            print("  [R] 轨迹已清除")
        elif key == ord('s') or key == ord('S'):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, combined)
            print(f"  [S] 截图已保存: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("定位已结束。")


if __name__ == "__main__":
    main()
