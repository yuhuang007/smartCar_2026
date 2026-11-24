# road/test_seg_ar_feed.py
# 读取官方 WebUI 提供的 AR 融合视频流进行语义分割测试
# 用法：
#   cd ~/Desktop/setupUI
#   python3 road/test_seg_ar_feed.py
#   或指定流地址：
#   python3 road/test_seg_ar_feed.py --url http://192.168.1.85:8080/ar_feed

import os
import sys
import argparse

import cv2
import numpy as np

# 相对路径兼容（在 setupUI/ 下运行）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road.seg_infer import SegModel
from road.seg_post import lane_offset_from_mask, overlay_mask
from road.nav_command import NavCommand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8080/ar_feed",
                        help="官方 WebUI AR 融合视频流地址")
    parser.add_argument("--save", type=str, default="road/test_output", help="结果保存目录")
    parser.add_argument("--max", type=int, default=300, help="最大处理帧数")
    args = parser.parse_args()

    seg = SegModel()
    seg.init()
    nav = NavCommand(dead_zone=20.0)

    os.makedirs(args.save, exist_ok=True)

    cap = cv2.VideoCapture(args.url)
    if not cap.isOpened():
        print(f"无法打开视频流: {args.url}")
        return

    frame_id = 0
    while frame_id < args.max:
        ret, frame = cap.read()
        if not ret:
            break

        mask = seg.predict_mask(frame, input_size=(512, 512), num_classes=2, thresh=0.5)
        offset = lane_offset_from_mask(mask, roi_y_ratio=0.5)
        vis = overlay_mask(frame, mask, alpha=0.45)

        cv2.putText(vis, f"offset={offset:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cmd = nav.decide(offset, fps=0.0)
        cv2.putText(vis, f"cmd={cmd}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out_path = os.path.join(args.save, f"arfeed_{frame_id:05d}.jpg")
        cv2.imwrite(out_path, vis)
        frame_id += 1

    cap.release()
    print("AR流测试完成，共处理帧:", frame_id, "结果保存在:", args.save)


if __name__ == "__main__":
    main()