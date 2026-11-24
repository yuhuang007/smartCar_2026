# -*- coding: utf-8 -*-
"""
地图标定脚本

功能：
  1. 打开摄像头，实时检测 ArUco 标记
  2. 使用标记中心点计算单应性矩阵（消除角点顺序歧义）
  3. 保存标定结果到 homography.npz

使用方法：
  python map_calibrate.py
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


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


def compute_homography(corners, ids):
    """
    从检测到的地面标记中心点计算单应性矩阵

    关键改进：使用标记中心而非角点，避免角点顺序歧义导致的巨大误差
    """
    src_pts = []  # 像素坐标（标记中心）
    dst_pts = []  # 世界坐标（标记中心）

    if ids is None:
        return None, None

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in GROUND_MARKER_POSITIONS:
            # 计算标记在图像中的中心点
            pts = corners[i].reshape(-1, 2)
            center_px = pts.mean(axis=0)

            # 对应的世界坐标中心点
            world_center = np.array(GROUND_MARKER_POSITIONS[marker_id], dtype=np.float32)

            src_pts.append(center_px.astype(np.float32))
            dst_pts.append(world_center)

    if len(src_pts) < 4:
        print(f"  [!] 只检测到 {len(src_pts)} 个地面标记，需要至少 4 个才能标定")
        return None, None

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    if H is None:
        return None, None

    # 计算重投影误差
    projected = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H)
    errors = np.linalg.norm(projected.reshape(-1, 2) - dst_pts, axis=1)
    mean_error = np.mean(errors)

    return H, mean_error


def draw_status(frame, detected_ids, calib_count, num_samples):
    """在画面上绘制检测状态"""
    y = 30
    n_detected = len([m for m in GROUND_MARKER_IDS if m in detected_ids])
    color = (0, 200, 0) if n_detected == 4 else (0, 0, 220)
    cv2.putText(frame, f"Ground markers: {n_detected}/4", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if calib_count > 0:
        cv2.putText(frame, f"Sampling... {calib_count}/{num_samples}",
                    (10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

    missing = [m for m in GROUND_MARKER_IDS if m not in detected_ids]
    if missing:
        cv2.putText(frame, f"Missing: {missing}", (10, y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    cv2.putText(frame, "SPACE=sample  Q=quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def main():
    print("=" * 55)
    print("  地图标定工具  (基于标记中心点)")
    print("=" * 55)
    print()
    print("操作步骤：")
    print("  1. 确保摄像头能看到全部 4 个地面参考标记")
    print("  2. 等待检测稳定后，按【空格键】采集")
    print("  3. 累计采集 5 次后自动保存标定结果")
    print("  4. 按【Q】随时退出")
    print()

    # 检查是否有摄像头内参校准文件
    if os.path.exists("camera_calibrate.npz"):
        data = np.load("camera_calibrate.npz")
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        print("[信息] 检测到摄像头内参校准文件，将自动校正镜头畸变")
    else:
        camera_matrix = None
        dist_coeffs = None
        print("[提示] 未检测到 camera_calibrate.npz，未做镜头畸变校正")
        print("       (建议先运行 camera_calibrate.py 进行摄像头校准)")
    print()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头！请检查 CAMERA_INDEX 设置。")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    detector = create_detector()

    NUM_SAMPLES = 5
    H_accumulated = None
    sample_count = 0
    total_error = 0.0

    print("摄像头已打开，开始检测...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[错误] 无法读取摄像头画面！")
            break

        # 如果有内参，先去畸变
        if camera_matrix is not None:
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        corners, ids = detect_markers(detector, frame)
        vis_frame = frame.copy()

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)

        found_ids = set(ids.flatten()) if ids is not None else set()
        draw_status(vis_frame, found_ids, sample_count, NUM_SAMPLES)

        cv2.imshow("Map Calibration - Press SPACE to sample", vis_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            n_detected = len([m for m in GROUND_MARKER_IDS if m in found_ids])
            if n_detected < 4:
                print(f"  [!] 需要检测到全部 4 个地面标记（当前 {n_detected}/4）")
                continue

            H, error = compute_homography(corners, ids)
            if H is not None:
                sample_count += 1
                if H_accumulated is None:
                    H_accumulated = H.astype(np.float64)
                else:
                    H_accumulated += H
                total_error += error
                print(f"  [{sample_count}/{NUM_SAMPLES}] 采集成功 "
                      f"(重投影误差: {error * 1000:.1f} mm)")

                if sample_count >= NUM_SAMPLES:
                    H_final = H_accumulated / NUM_SAMPLES
                    H_final /= H_final[2, 2]
                    avg_error = total_error / NUM_SAMPLES
                    np.savez(HOMOGRAPHY_FILE, H=H_final)
                    print()
                    print("=" * 55)
                    print(f"  标定完成！")
                    print(f"  平均重投影误差: {avg_error * 1000:.1f} mm")
                    print(f"  已保存到: {HOMOGRAPHY_FILE}")
                    if camera_matrix is None:
                        print()
                        print("  [建议] 误差仍偏高时，可运行 camera_calibrate.py")
                        print("         校正镜头畸变后重新标定")
                    print("=" * 55)
                    break
            else:
                print("  [!] 计算失败")

        elif key == ord('q') or key == ord('Q'):
            print("退出标定")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
