# road/test_seg_offline.py
# 离线测试语义分割（不依赖比赛共享内存）
# 用法：
#   cd ~/Desktop/setupUI
#   python3 road/test_seg_offline.py
#   或指定图片：
#   python3 road/test_seg_offline.py --image ../datalane/Images/frame_000001.jpg
#   或指定图片目录：
#   python3 road/test_seg_offline.py --dir ../datalane/Images

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
    parser.add_argument("--image", type=str, default=None, help="单张图片路径")
    parser.add_argument("--dir", type=str, default=None, help="图片目录（批量测试）")
    parser.add_argument("--thresh", type=float, default=0.5, help="mask阈值")
    parser.add_argument("--roi", type=float, default=0.5, help="ROI起始比例(0~1)")
    parser.add_argument("--save", type=str, default="road/test_output", help="结果保存目录")
    args = parser.parse_args()

    seg = SegModel()
    seg.init()
    nav = NavCommand(dead_zone=20.0)

    os.makedirs(args.save, exist_ok=True)

    paths = []
    if args.image:
        paths = [args.image]
    elif args.dir:
        for name in sorted(os.listdir(args.dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                paths.append(os.path.join(args.dir, name))
    else:
        # 默认用你给的数据集目录
        default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datalane", "Images"))
        if os.path.isdir(default_dir):
            for name in sorted(os.listdir(default_dir)):
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    paths.append(os.path.join(default_dir, name))
        else:
            print("未找到图片，请用 --image 或 --dir 指定")
            return

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"读取失败: {p}")
            continue

        _dbg_dir = args.save
        _base = os.path.splitext(os.path.basename(p))[0]

        # 1) 先拿 raw prob
        prob = seg.predict_prob(img, input_size=(384, 384), num_classes=2)

        # 2) 兼容 prob shape
        if prob.ndim == 3:
            prob2d = prob[0] if prob.shape[0] == 1 else (prob[1] if prob.shape[0] >= 2 else prob[0])
        elif prob.ndim == 2:
            prob2d = prob
        else:
            prob2d = prob

        # 3) 保存 raw prob（16bit，避免归一化丢失信息）
        _prob16_path = os.path.join(_dbg_dir, f"{_base}_prob16.png")
        prob2d_u16 = np.clip(prob2d * 1000, 0, 65535).astype(np.uint16)  # 放大后保存
        cv2.imwrite(_prob16_path, prob2d_u16)

        # 4) 保存伪彩色（方便肉眼看分布）
        _prob_color_path = os.path.join(_dbg_dir, f"{_base}_prob_color.png")
        prob2d_norm = np.clip((prob2d - prob2d.min()) / (max(1e-6, prob2d.max() - prob2d.min())), 0, 1)
        prob2d_u8 = (prob2d_norm * 255).astype(np.uint8)
        prob_color = cv2.applyColorMap(prob2d_u8, cv2.COLORMAP_JET)
        cv2.imwrite(_prob_color_path, prob_color)

        print(f"[DBG] saved: {_prob16_path}, {_prob_color_path}")
        print(f"[DBG] prob2d min={prob2d.min():.6f}, max={prob2d.max():.6f}, mean={prob2d.mean():.6f}")

        # 5) 阈值扫描（打印不同阈值下白色像素占比）
        for th in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
            cnt = int((prob2d > th).sum())
            print(f"[DBG] prob>{th:.2f} count={cnt}")

        # 6) 生成 mask：这里先用 mask_inv（因为当前 prob 全黑、我们怀疑道路是低值区域）
        mask = seg.predict_mask(img, input_size=(384, 384), num_classes=2, thresh=0.2, use_sigmoid=False)
        mask_inv = 1 - mask

        cv2.imwrite(os.path.join(_dbg_dir, f"{_base}_mask.png"), (mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(_dbg_dir, f"{_base}_mask_inv.png"), (mask_inv * 255).astype(np.uint8))
        print("[DBG] saved mask/mask_inv")

        # 7) 视觉结果先用 mask_inv（待确认后再固定）
        mask_for_vis = mask_inv
        offset = lane_offset_from_mask(mask_for_vis, roi_y_ratio=args.roi)
        vis = overlay_mask(img, mask_for_vis, alpha=0.45)
        
        cv2.putText(vis, f"offset={offset:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cmd = nav.decide(offset, fps=0.0)
        cv2.putText(vis, f"cmd={cmd}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(args.save, f"{base}_seg.jpg")
        cv2.imwrite(out_path, vis)
        print(f"{p} -> {out_path}, offset={offset:.1f}, cmd={cmd}")

    print("离线测试完成，结果保存在:", args.save)


if __name__ == "__main__":
    main()