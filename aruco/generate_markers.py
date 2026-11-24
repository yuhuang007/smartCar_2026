# -*- coding: utf-8 -*-
"""
生成 ArUco 标记图片（4m 高度 / 4m x 5m 地图适用）

地面标记: 25cm x 25cm
车载标记: 15cm x 15cm
"""

import cv2
import numpy as np
import os

def generate_marker(dictionary, marker_id, size_px=600, border_px=80):
    """生成单个 ArUco 标记，带白色边框"""
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    img_with_border = cv2.copyMakeBorder(
        img, border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT, value=255
    )
    return img_with_border

def add_label(img, text, position=(60, 50)):
    """在图片上添加文字标签"""
    labeled = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    cv2.putText(labeled, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    return labeled

def main():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    output_dir = "markers"
    os.makedirs(output_dir, exist_ok=True)

    markers = [
        (0,  "Ground-TopLeft(0,0)",       "25cm"),
        (1,  "Ground-TopRight(4m,0)",     "25cm"),
        (2,  "Ground-BottomLeft(0,5m)",   "25cm"),
        (3,  "Ground-BottomRight(4m,5m)", "25cm"),
        (10, "Car marker",                "15cm"),
    ]

    print("=" * 60)
    print("  ArUco Marker Generator (4m height / 4x5m map)")
    print("=" * 60)

    for marker_id, description, print_size in markers:
        img = generate_marker(dictionary, marker_id)
        labeled = add_label(img, f"ID={marker_id}  {print_size}")
        filepath = os.path.join(output_dir, f"marker_{marker_id}.png")
        cv2.imwrite(filepath, labeled)
        pure_path = os.path.join(output_dir, f"marker_{marker_id}_pure.png")
        cv2.imwrite(pure_path, img)
        print(f"  [OK] ID={marker_id:2d}  {description}  ->  {filepath}")

    print()
    print("Print these markers and place them on the ground:")
    print()
    print("  +-----------------------------------------+")
    print("  | marker_0(0,0)         marker_1(4m,0)    |")
    print("  |      *---------------------*             |")
    print("  |      |                     |             |")
    print("  |      |     Map 4m x 5m     |   5m        |")
    print("  |      |                     |             |")
    print("  |      *---------------------*             |")
    print("  | marker_2(0,5m)         marker_3(4m,5m)  |")
    print("  +-----------------------------------------+")
    print("                  4m")
    print()
    print("  marker_10 -> stick on top of the smart car")
    print()
    print("Print settings:")
    print("  - Ground markers: 25cm x 25cm each")
    print("  - Car marker:     15cm x 15cm")
    print("  - Use laser printer, high contrast black/white")
    print("  - Stick on cardboard to keep flat")

if __name__ == "__main__":
    main()