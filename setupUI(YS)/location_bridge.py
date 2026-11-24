# -*- coding: utf-8 -*-
"""
ArUco定位中继脚本(多线程版)
功能: 运行ArUco定位, 将实车位置通过UDP发送到9005(引擎)和9006(main_control.py)
用法: python location_bridge.py
作用: 平替官方定位模块, 比赛时换成官方定位即可
"""
import cv2
import numpy as np
import sys
import os
import time
import json
import socket
import threading

# 添加aruco模块路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aruco"))
sys.path.insert(0, r"C:\Users\Administrator\Desktop\car\aruco")  # 本机aruco路径

try:
    from config import *
except ImportError:
    ARUCO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aruco")
    sys.path.insert(0, ARUCO_DIR)
    from config import *

# 复用track.py的函数
try:
    from track import create_detector, detect_markers, pixel_to_world, compute_heading, SmoothingFilter
except ImportError:
    print("[错误] 无法导入track.py，请确保aruco目录在路径中")
    sys.exit(1)


def main():
    ORANGEPI_IP = "10.101.221.232"
    ENGINE_ADDR = (ORANGEPI_IP, 9005)
    CONTROL_ADDR = (ORANGEPI_IP, 9006)

    # --- 加载标定文件 ---
    aruco_dir = os.path.dirname(os.path.abspath(__file__))
    homography_path = os.path.join(aruco_dir, "..", "aruco", HOMOGRAPHY_FILE)
    if not os.path.exists(homography_path):
        homography_path = os.path.join(r"C:\Users\Administrator\Desktop\car\aruco", HOMOGRAPHY_FILE)
    if not os.path.exists(homography_path):
        print(f"[错误] 找不到标定文件 '{HOMOGRAPHY_FILE}'！")
        sys.exit(1)
    data = np.load(homography_path)
    H = data["H"]
    print(f"已加载标定文件: {homography_path}")

    # --- 加载摄像头内参 ---
    camera_matrix = None
    dist_coeffs = None
    cam_calib_path = os.path.join(os.path.dirname(homography_path), "camera_calibrate.npz")
    if os.path.exists(cam_calib_path):
        cam_data = np.load(cam_calib_path)
        camera_matrix = cam_data["camera_matrix"]
        dist_coeffs = cam_data["dist_coeffs"]
        print("已加载摄像头内参")

    # --- 共享状态 ---
    location = {"x": None, "z": None, "yaw": None, "seq": 0, "lock": threading.Lock()}
    running = {"flag": True}

    # --- UDP发送线程(持续发送最新位置, 不等检测) ---
    def udp_sender():
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        seq = 0
        last_sent = None
        while running["flag"]:
            with location["lock"]:
                x, z, yaw = location["x"], location["z"], location["yaw"]
            if x is not None and (x, z, yaw) != last_sent:
                seq += 1
                payload = {
                    "type": "robot_position",
                    "is_sim": False,
                    "pos": [round(float(x), 6), 0.0, round(float(z), 6)],
                    "euler": [0.0, round(float(yaw), 6), 0.0],
                    "seq": seq,
                    "timestamp": time.time(),
                }
                data = json.dumps(payload).encode("utf-8")
                try:
                    udp_sock.sendto(data, ENGINE_ADDR)
                    udp_sock.sendto(data, CONTROL_ADDR)
                except Exception:
                    pass
                last_sent = (x, z, yaw)
                if seq % 10 == 0:
                    print(f"[ArUco] ({x:.2f}, {z:.2f})m  {yaw:.1f}°  seq={seq}")
            time.sleep(0.005)  # 200fps发送频率上限
        udp_sock.close()

    # --- ArUco检测线程(持续检测, 更新共享位置) ---
    def aruco_detector():
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[错误] 无法打开摄像头！")
            running["flag"] = False
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        detector = create_detector()
        pos_filter = SmoothingFilter(alpha=0.4)
        # ★航向滤波: 自定义实现, 含跳变拒绝+角度环绕
        _smoothed_heading = None

        def filter_heading(raw_h, alpha=0.15, max_jump=45):
            """航向角滤波: 低通+跳变拒绝+角度环绕
            alpha=0.15: 15%新值(强平滑, 防ArUco角点误排序导致跳变)
            max_jump=45°: 单帧跳变超过45°视为误检, 拒绝
            """
            nonlocal _smoothed_heading
            if _smoothed_heading is None:
                _smoothed_heading = raw_h
                return raw_h
            # 角度环绕: 取最短角差
            diff = raw_h - _smoothed_heading
            while diff > 180: diff -= 360
            while diff < -180: diff += 360
            # 跳变拒绝: |diff|>45° → ArUco角点误排序, 丢弃
            if abs(diff) > max_jump:
                return _smoothed_heading  # 保持旧值
            # 低通滤波
            _smoothed_heading += alpha * diff
            # 归一化到[-180, 180]
            while _smoothed_heading > 180: _smoothed_heading -= 360
            while _smoothed_heading < -180: _smoothed_heading += 360
            return _smoothed_heading

        while running["flag"]:
            ret, frame = cap.read()
            if not ret:
                continue
            if camera_matrix is not None:
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
            corners, ids = detect_markers(detector, frame)
            if ids is not None:
                id_list = ids.flatten().tolist()
                if CAR_MARKER_ID in id_list:
                    idx = id_list.index(CAR_MARKER_ID)
                    marker_corners = corners[idx]
                    center_px = marker_corners.reshape(-1, 2).mean(axis=0)
                    raw_x, raw_z = pixel_to_world(H, center_px[0], center_px[1])
                    raw_heading = compute_heading(marker_corners)
                    smoothed = pos_filter.update([raw_x, raw_z])
                    h = filter_heading(raw_heading)  # ★用自定义航向滤波
                    with location["lock"]:
                        location["x"] = smoothed[0]
                        location["z"] = smoothed[1]
                        location["yaw"] = h
                else:
                    pos_filter.reset()
                    _smoothed_heading = None  # ★重置航向滤波
        cap.release()

    # --- 启动线程 ---
    print("=" * 55)
    print("  ArUco定位中继已启动(多线程版)")
    print(f"  定位数据 → UDP 9005(引擎) + 9006(main_control)")
    print("  Ctrl+C退出")
    print("=" * 55)

    t_detect = threading.Thread(target=aruco_detector, daemon=True)
    t_send = threading.Thread(target=udp_sender, daemon=True)
    t_detect.start()
    t_send.start()

    try:
        while running["flag"]:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
        running["flag"] = False
        t_detect.join(timeout=2)
        t_send.join(timeout=2)
    print("定位中继已停止。")


if __name__ == "__main__":
    main()
