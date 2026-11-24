# -*- coding: utf-8 -*-
"""
摄像头内参标定脚本（可选但推荐）

功能：
  使用棋盘格标定板校正摄像头镜头畸变
  标定后会生成 camera_calibrate.npz，供后续脚本自动使用

使用方法：
  1. 打印一张棋盘格图案（可以用搜索引擎搜 "chessboard calibration pattern" 打印一张 A4 的）
  2. 运行 python camera_calibrate.py
  3. 手持棋盘格在摄像头前不断变换角度和位置
  4. 按空格键采集，共采集 15~20 张
  5. 按 Q 完成标定

棋盘格建议：
  - 内角点数: 7x5（横向 7 个角点，纵向 5 个角点）
  - 格子边长: 实际打印后用尺子量一个格子的大小（米），填入下面的 SQUARE_SIZE
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# ==================== 棋盘格参数 ====================
# 棋盘格内部角点数量（注意：不是格子数量，是内部交叉点数量）
# 例如: 8列6行的棋盘格，内部角点是 7x5
CHESSBOARD_SIZE = (7, 5)

# 每个棋盘格方格的实际边长（单位：米）
# 打印后用尺子量一下，填入实际值
SQUARE_SIZE = 0.025  # 2.5cm，根据实际打印尺寸修改


def main():
    print("=" * 55)
    print("  摄像头内参标定工具")
    print("=" * 55)
    print()
    print("准备一张棋盘格打印纸，在摄像头前变换角度和距离：")
    print("  - 按【空格键】采集一张")
    print("  - 采集 15~20 张不同角度后，按【Q】完成标定")
    print()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头！")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # 准备棋盘格的世界坐标
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                            0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    obj_points = []  # 3D 世界坐标
    img_points = []  # 2D 图像坐标

    img_shape = None
    sample_count = 0

    print("摄像头已打开，开始检测棋盘格...")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[错误] 无法读取摄像头画面！")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        # 检测棋盘格
        found, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE, None,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            # 亚像素精细定位
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, found)
            cv2.putText(vis, "Chessboard DETECTED - Press SPACE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        else:
            cv2.putText(vis, "No chessboard found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

        cv2.putText(vis, f"Samples: {sample_count}/15~20", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(vis, "SPACE=sample  Q=calibrate", (10, vis.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Camera Calibration", vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if found:
                img_shape = gray.shape[::-1]
                obj_points.append(objp.copy())
                img_points.append(corners)
                sample_count += 1
                print(f"  [{sample_count}] 采集成功")
            else:
                print("  [!] 未检测到棋盘格，请调整位置")

        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if sample_count < 5:
        print(f"\n[错误] 采集不足（仅 {sample_count} 张），请至少采集 5 张")
        sys.exit(1)

    print(f"\n正在使用 {sample_count} 张图片进行标定...")

    # 执行标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )

    # 计算重投影误差
    total_error = 0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i],
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    mean_error = total_error / len(obj_points)

    # 保存
    np.savez("camera_calibrate.npz",
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)

    print()
    print("=" * 55)
    print("  摄像头标定完成！")
    print(f"  重投影误差: {mean_error * 1000:.2f} mm")
    print(f"  已保存到: camera_calibrate.npz")
    print()
    print("  摄像头内参矩阵:")
    print(f"    {camera_matrix[0]}")
    print(f"    {camera_matrix[1]}")
    print(f"    {camera_matrix[2]}")
    print()
    print("  畸变系数:")
    print(f"    {dist_coeffs.flatten()}")
    print()
    print("  下次运行 map_calibrate.py 时会自动加载此文件校正畸变")
    print("=" * 55)


if __name__ == "__main__":
    main()
