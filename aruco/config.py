# -*- coding: utf-8 -*-
"""
配置文件 - 4米高摄像头 / 4m x 5m 地图
"""

# ==================== 地图参数 ====================
MAP_WIDTH  = 4.0
MAP_HEIGHT = 5.0

# ==================== 摄像头高度 ====================
CAMERA_HEIGHT_M = 3.17

# ==================== 标定板参数 ====================
GROUND_MARKER_SIZE = 0.25
CAR_MARKER_SIZE = 0.15

# ==================== ArUco 标记 ID 分配 ====================
GROUND_MARKER_IDS = [0, 1, 2, 3]
CAR_MARKER_ID = 10

# 地面标记世界坐标 (x, z)，单位：米
GROUND_MARKER_POSITIONS = {
    0: (0.0, MAP_HEIGHT),       # (0, 5)
    1: (MAP_WIDTH, MAP_HEIGHT), # (4, 5)
    2: (0.0, 0.0),              # (0, 0) origin
    3: (MAP_WIDTH, 0.0),        # (4, 0)
}

# ==================== 相机参数 ====================
CAMERA_INDEX = 0
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

# ==================== 文件路径 ====================
HOMOGRAPHY_FILE = "homography.npz"