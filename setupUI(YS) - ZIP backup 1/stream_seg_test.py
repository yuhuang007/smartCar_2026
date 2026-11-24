"""
视频流语义分割 + 边缘检测 测试脚本

可以从Windows远程连接板卡的视频流进行测试，不需要在板卡上运行。

两种使用方式:
  方式1 (Windows远程调试): 读取HTTP视频流 → 用OpenCV做边缘检测 → 显示
  方式2 (板卡本地运行): 读取共享内存 → NPU语义分割 + 边缘检测 → 显示

用法:
  # Windows远程测试 (只需改IP地址)
  python stream_seg_test.py --mode http --url http://192.168.1.85:8080/ar_feed

  # 板卡本地测试 (需要RKNN环境)
  python stream_seg_test.py --mode shm

  # 用原生流(无AR)测试赛道识别
  python stream_seg_test.py --mode http --url http://192.168.1.85:8080/video_feed
"""

import cv2
import numpy as np
import time
import argparse
import struct

# ============ 边缘检测模块 ============
class EdgeDetector:
    """
    边缘检测器：用于辅助赛道线识别

    技术方法:
    1. 高斯模糊去噪
    2. Canny边缘检测
    3. ROI掩膜（只保留图像中下部=车前方道路区域）
    4. 霍夫变换提取直线（车道线）
    5. 融合语义分割mask做路径规划
    """

    def __init__(self,
                 canny_low=50,
                 canny_high=150,
                 roi_top_ratio=0.5,
                 hough_threshold=50,
                 hough_min_line_length=50,
                 hough_max_line_gap=150):
        """
        参数:
            canny_low/high: Canny边缘检测的高低阈值
            roi_top_ratio: ROI区域起始位置（0.5=图像下半部分）
            hough_threshold: 霍夫变换投票阈值
            hough_min_line_length: 最小线段长度
            hough_max_line_gap: 最大线段间隙
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.roi_top_ratio = roi_top_ratio
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap

    def detect(self, frame):
        """
        完整边缘检测流程

        参数:
            frame: BGR格式图像

        返回:
            edge_vis: 可视化结果（原图+边缘+车道线）
            edges: Canny边缘二值图
            lines: 检测到的车道线列表 [(x1,y1,x2,y2), ...]
        """
        h, w = frame.shape[:2]

        # 1. 转灰度 + 高斯模糊
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Canny边缘检测
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # 3. ROI掩膜：只保留图像中下部（车前方道路区域）
        roi_mask = np.zeros_like(edges)
        roi_top = int(h * self.roi_top_ratio)
        # 梯形ROI：上窄下宽
        roi_vertices = np.array([[
            (int(w * 0.1), h),           # 左下
            (int(w * 0.35), roi_top),    # 左上
            (int(w * 0.65), roi_top),    # 右上
            (int(w * 0.9), h),           # 右下
        ]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        # 4. 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        # 5. 可视化
        edge_vis = frame.copy()

        # 画ROI区域（半透明绿色）
        roi_overlay = edge_vis.copy()
        cv2.fillPoly(roi_overlay, roi_vertices, (0, 255, 0))
        edge_vis = cv2.addWeighted(edge_vis, 0.85, roi_overlay, 0.15, 0)

        # 画Canny边缘（红色）
        edge_vis[masked_edges > 0] = (0, 0, 255)

        # 画车道线（黄色粗线）
        detected_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                detected_lines.append((x1, y1, x2, y2))
                cv2.line(edge_vis, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # 标注信息
        cv2.putText(edge_vis, f"Edges: {len(detected_lines)} lines",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return edge_vis, masked_edges, detected_lines


# ============ 蓝色箭头导航模块 ============
class ArrowNavigator:
    """
    蓝色箭头导航器（针对本赛道: 地面用蓝色发光箭头/人字形铺成）

    技术方法:
    1. HSV颜色分割提取蓝色箭头像素（比边缘检测鲁棒得多）
    2. 把车前方分近/远两条带，各算蓝色质心:
       - 近带质心 = 当前车应处的横向位置
       - 远带质心 = 前方赛道走向（弯道提前转）
    3. 输出融合偏移量，代替不可靠的边缘检测
    """

    def __init__(self,
                 lower_hsv=(85, 50, 80),
                 upper_hsv=(125, 255, 255),
                 near_weight=0.6,
                 min_pixels=80,
                 roi_margin_x=0.0,
                 min_component_area=None):
        """
        参数:
            lower_hsv/upper_hsv: 蓝色箭头的HSV阈值范围（根据实际颜色微调）
            near_weight: 近带权重（越大越相信当前位置而非前方）
            min_pixels: 每带最少蓝色像素数（低于此视为无效）
            roi_margin_x: 左右边缘剔除比例(0.08=左右各8%不检测, 排除场地边缘/光线反射的干扰蓝色)
            min_component_area: 连通域最小面积(低于此的零星干扰蓝色被剔除, None=用min_pixels)
        """
        self.lower = np.array(lower_hsv, dtype=np.uint8)
        self.upper = np.array(upper_hsv, dtype=np.uint8)
        self.near_weight = near_weight
        self.min_pixels = min_pixels
        self.roi_margin_x = roi_margin_x
        self.min_component_area = min_component_area if min_component_area is not None else min_pixels
        self.last_offset = 0.0
        self.alpha = 0.92  # 平滑系数(0.85滞后15%, 0.92滞后8%高速更跟手)
        self.kernel = np.ones((5, 5), np.uint8)

    def _band_centroid(self, mask, w):
        """计算一条带内蓝色像素的横向质心，返回 (cx_norm, valid)"""
        xs = np.where(mask > 0)[1]
        if len(xs) < self.min_pixels:
            return 0.0, False
        cx = float(np.mean(xs))
        return (cx / w) - 0.5, True

    def navigate(self, frame):
        """
        蓝色箭头导航（远近双带版）

        近带(0.65~1.0h): 横向偏差修正(当前车在赛道的位置)
        远带(0.40~0.65h): 前方走向预判(弯道提前转)
        offset = w_near×近带 + w_far×远带

        返回:
            offset: 转向偏移量 (-0.5~+0.5, 正=右, 负=左)
            vis: 可视化图
            valid: 是否检测到有效箭头
            info: dict 详细信息
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        # 形态学去噪 + 连通
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # ★ROI限制: 剔除左右边缘(场地边缘/光线反射的干扰蓝色, 解决质心系统性偏移)
        if self.roi_margin_x > 0:
            mx = int(w * self.roi_margin_x)
            if mx > 0:
                mask[:, :mx] = 0
                mask[:, w - mx:] = 0

        # ★连通域过滤: 剔除零星干扰蓝色, 只保留够大的赛道连通域
        _num_cc, _labels, _stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if _num_cc > 1:
            _clean = np.zeros_like(mask)
            for _i in range(1, _num_cc):
                if _stats[_i, cv2.CC_STAT_AREA] >= self.min_component_area:
                    _clean[_labels == _i] = 255
            mask = _clean

        # ★Pure Pursuit: 逐行求蓝色中心(中位数), 多项式拟合赛道中线
        far_top = int(h * 0.40)    # 远带上边界
        near_top = int(h * 0.65)   # 近带上边界

        # 逐行采样蓝色中心点(中位数比均值鲁棒, 排除离群蓝色)
        center_points = []  # [(y, cx), ...]
        for _yy in range(far_top, h, 4):
            _xs = np.where(mask[_yy] > 0)[0]
            if len(_xs) >= 10:
                center_points.append((_yy, float(np.median(_xs))))

        # 像素统计(info用, 两个分支共用)
        near_xs = np.where(mask[near_top:, :] > 0)[1]
        far_xs = np.where(mask[far_top:near_top, :] > 0)[1]

        if len(center_points) >= 3:
            # 多项式拟合中线 cx=f(y): ≥5点用二次(弯道), 否则一次(直道)
            _pts_y = np.array([p[0] for p in center_points])
            _pts_x = np.array([p[1] for p in center_points])
            _deg = 2 if len(center_points) >= 5 else 1
            _coeffs = np.polyfit(_pts_y, _pts_x, _deg)
            # 从拟合中线取近点(当前位置)和远点(预瞄点)
            near_cx = float(np.polyval(_coeffs, int(h * 0.85)))
            far_cx = float(np.polyval(_coeffs, int(h * 0.50)))
            near_offset = (near_cx / w) - 0.5
            far_offset = (far_cx / w) - 0.5
            near_valid = True
            far_valid = True
        else:
            # 拟合点不足, 回退到质心法
            near_valid = len(near_xs) >= self.min_pixels
            if near_valid:
                near_offset = (float(np.median(near_xs)) / w) - 0.5
            else:
                near_offset = self.last_offset
            far_valid = len(far_xs) >= max(self.min_pixels // 2, 30)
            if far_valid:
                far_offset = (float(np.median(far_xs)) / w) - 0.5
            else:
                far_offset = near_offset

        # 融合: 近带修正横向偏差, 远带预判前方走向
        w_near = self.near_weight
        w_far = 1.0 - w_near
        if near_valid and far_valid:
            # ★岔路检测: 近带偏移远小于远带(近带<远带×30%)且远带偏移>0.08
            #   =远带检测到的是岔路不是弯道, 忽略远带防误转
            if abs(far_offset) > 0.08 and abs(near_offset) < abs(far_offset) * 0.3:
                raw_offset = near_offset  # 忽略远带
            else:
                raw_offset = w_near * near_offset + w_far * far_offset
        elif near_valid:
            raw_offset = near_offset
        elif far_valid:
            raw_offset = far_offset
        else:
            raw_offset = self.last_offset
            near_offset = self.last_offset
            far_offset = self.last_offset

        valid = near_valid or far_valid

        # 平滑
        offset = self.last_offset * (1 - self.alpha) + raw_offset * self.alpha
        offset = max(-0.5, min(0.5, offset))
        self.last_offset = offset

        # 可视化
        vis = frame.copy()
        blue = np.zeros_like(frame)
        blue[mask > 0] = (255, 128, 0)
        vis = cv2.addWeighted(vis, 0.7, blue, 0.3, 0)
        # 双带ROI边界
        cv2.line(vis, (0, far_top), (w, far_top), (0, 200, 200), 1)    # 远带上边界(黄)
        cv2.line(vis, (0, near_top), (w, near_top), (0, 200, 0), 1)    # 近带上边界(绿)
        # 逐行中心点
        for yy in range(far_top, h, 20):
            rx = np.where(mask[yy] > 0)[0]
            if len(rx) > 5:
                cx_row = int(np.median(rx))
                cv2.circle(vis, (cx_row, yy), 2, (0, 255, 0), -1)
        # 近带质心(绿) + 远带质心(黄)
        if near_valid:
            cv2.circle(vis, (int((near_offset + 0.5) * w), int(h * 0.82)),
                       8, (0, 255, 0), -1)
        if far_valid:
            cv2.circle(vis, (int((far_offset + 0.5) * w), int(h * 0.52)),
                       8, (0, 255, 255), -1)
        # 融合offset中心线(红)
        tgt_x = int((offset + 0.5) * w)
        cv2.line(vis, (tgt_x, far_top), (tgt_x, h), (0, 0, 255), 2)
        cv2.putText(vis, f"ARROW off={offset:.3f} {'OK' if valid else 'LOST'} "
                    f"near={near_offset:+.3f}{'!' if not near_valid else ''} "
                    f"far={far_offset:+.3f}{'!' if not far_valid else ''}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        info = {
            "raw_offset": raw_offset,
            "near_offset": near_offset,
            "far_offset": far_offset,
            "near_valid": near_valid,
            "far_valid": far_valid,
            "valid": valid,
            "pixels": int(np.count_nonzero(mask)),
            "near_pixels": int(len(near_xs)),
            "far_pixels": int(len(far_xs)),
        }
        return offset, vis, valid, info


# ============ 路径规划模块 ============
class PathPlanner:
    """
    路径规划器：融合语义分割 + 边缘检测结果

    决策逻辑:
    1. 语义分割提供赛道区域（大面积）
    2. 边缘检测提供车道线（精确边界）
    3. 两者融合 → 计算最优行驶路径中心
    4. 输出转向偏移量给串口控制
    """

    def __init__(self):
        self.last_offset = 0.0  # 上一帧偏移量（用于平滑滤波）
        self.alpha = 0.7  # 平滑系数：越大越相信当前帧

    def plan(self, seg_mask, edge_lines, frame_shape):
        """
        路径规划

        参数:
            seg_mask: 语义分割mask (0=背景, 1=赛道)
            edge_lines: 边缘检测到的车道线列表
            frame_shape: (height, width)

        返回:
            offset: 转向偏移量 (-0.5~+0.5, 正=右转, 负=左转)
            center_point: (cx, cy) 路径中心点坐标
            vis_info: dict, 可视化信息
        """
        h, w = frame_shape[:2]

        # 方法1: 从分割mask提取中心
        seg_offset = 0.0
        seg_center = (w // 2, h // 2)
        seg_valid = False   # 分割是否有效（识别到足够赛道像素）

        if seg_mask is not None and seg_mask.size > 0:
            # 取图像底部1/3区域
            roi = seg_mask[int(h * 0.6):, :]
            track_pixels = np.where(roi == 1)

            if len(track_pixels[0]) > 50:
                seg_cx = np.mean(track_pixels[1])
                seg_cy = np.mean(track_pixels[0]) + int(h * 0.6)
                seg_offset = (seg_cx / w) - 0.5
                seg_center = (int(seg_cx), int(seg_cy))
                seg_valid = True

        # 方法2: 从边缘检测车道线提取中心
        edge_offset = 0.0
        edge_center = (w // 2, h // 2)
        edge_valid = False   # 边缘检测是否有效（提取到车道中心）
        is_junction = False  # 是否叉路口

        if edge_lines and len(edge_lines) >= 2:
            # 分离左右车道线
            left_lines = []
            right_lines = []

            for x1, y1, x2, y2 in edge_lines:
                # 计算斜率
                if abs(x2 - x1) < 5:
                    continue  # 垂直线跳过
                slope = (y2 - y1) / (x2 - x1)

                # 斜率为负=左车道线（图像坐标系y向下）；阈值0.4更严格，滤掉近水平的路口横线
                if slope < -0.4:
                    left_lines.append((x1, y1, x2, y2))
                elif slope > 0.4:
                    right_lines.append((x1, y1, x2, y2))

            # 计算左右车道线中心
            left_cx = None
            right_cx = None

            if left_lines:
                left_cx = np.mean([(l[0] + l[2]) / 2 for l in left_lines])
            if right_lines:
                right_cx = np.mean([(r[0] + r[2]) / 2 for r in right_lines])

            if left_cx is not None and right_cx is not None:
                # 两条线都有
                lane_gap = abs(right_cx - left_cx) / w   # 左右线间距（归一化）
                if lane_gap > 0.7:
                    # 左右线跨度过大 → 叉路口（横向路口线干扰）→ 保持直行通过，不跟横线走斜
                    edge_offset = self.last_offset
                    edge_center = (w // 2, int(h * 0.75))
                    edge_valid = True
                    is_junction = True
                else:
                    # 正常双车道线 → 中心取平均
                    edge_cx = (left_cx + right_cx) / 2
                    edge_offset = (edge_cx / w) - 0.5
                    edge_center = (int(edge_cx), int(h * 0.75))
                    edge_valid = True
            elif left_cx is not None:
                # 只有左线 → 中心在左线右侧一定距离
                edge_cx = left_cx + w * 0.2
                edge_offset = (edge_cx / w) - 0.5
                edge_center = (int(edge_cx), int(h * 0.75))
                edge_valid = True
            elif right_cx is not None:
                # 只有右线 → 中心在右线左侧一定距离
                edge_cx = right_cx - w * 0.2
                edge_offset = (edge_cx / w) - 0.5
                edge_center = (int(edge_cx), int(h * 0.75))
                edge_valid = True

        # 融合两种方法的偏移量（根据有效性动态分配权重）
        # 关键: 分割失效时不能用 seg=0 拉低整体，应完全依赖边缘检测
        if seg_valid and edge_valid:
            # 两者都有效 → 分割提供大区域主导，边缘微调
            fused_offset = seg_offset * 0.6 + edge_offset * 0.4
        elif edge_valid:
            # 只有边缘有效（分割失效）→ 全靠边缘检测
            fused_offset = edge_offset
        elif seg_valid:
            # 只有分割有效 → 全靠分割
            fused_offset = seg_offset
        else:
            # 都无效 → 保持上一帧偏移（避免突变回正）
            fused_offset = self.last_offset

        # 平滑滤波（防止抖动）
        smoothed_offset = self.last_offset * (1 - self.alpha) + fused_offset * self.alpha
        self.last_offset = smoothed_offset

        # 限制范围
        smoothed_offset = max(-0.5, min(0.5, smoothed_offset))

        vis_info = {
            "seg_offset": seg_offset,
            "edge_offset": edge_offset,
            "fused_offset": fused_offset,
            "smoothed_offset": smoothed_offset,
            "seg_center": seg_center,
            "edge_center": edge_center,
            "is_junction": is_junction,
        }

        return smoothed_offset, seg_center, vis_info


# ============ HTTP视频流读取器 ============
class HTTPStreamReader:
    """
    通过HTTP读取板卡视频流
    适用于Windows远程调试

    用法:
        reader = HTTPStreamReader("http://192.168.1.85:8080/ar_feed")
        frame = reader.read()
    """

    def __init__(self, url):
        self.url = url
        self.cap = None
        self._connect()

    def _connect(self):
        print(f"[HTTP] 连接视频流: {self.url}")
        self.cap = cv2.VideoCapture(self.url)
        if self.cap.isOpened():
            print(f"[HTTP] 连接成功! 分辨率: "
                  f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        else:
            print(f"[HTTP] 连接失败! 请检查:")
            print(f"  1. 板卡是否已启动setup_webui")
            print(f"  2. 地址是否正确: {self.url}")
            print(f"  3. 网络是否通: ping {self.url.split('//')[1].split(':')[0]}")

    def read(self):
        """读取一帧"""
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()


# ============ SHM视频流读取器（板卡本地用） ============
class SHMVideoReader:
    """
    通过共享内存读取X-Verse引擎视频帧
    适用于板卡本地运行（延迟最低）

    与 ar_receiver.py / main_control.py 中的 ARVideoReader 逻辑相同
    """

    def __init__(self, shm_name="shm_ar_video", header_size=16):
        from multiprocessing import shared_memory, resource_tracker
        self.shm_name = shm_name
        self.header_size = header_size
        self.shm = None
        self.last_fid = 0
        self._shm_module = shared_memory
        self._resource_tracker = resource_tracker

    def connect(self, max_wait=30):
        print(f"[SHM] 等待X-Verse引擎...")
        for i in range(max_wait):
            try:
                self.shm = self._shm_module.SharedMemory(name=self.shm_name)
                try:
                    self._resource_tracker.unregister('/' + self.shm_name, 'shared_memory')
                except:
                    pass
                print("[SHM] 连接成功!")
                return True
            except FileNotFoundError:
                time.sleep(1)
                print(f"[SHM] 等待中... ({i+1}/{max_wait})")
        print("[SHM] 连接超时!")
        return False

    def read(self):
        if self.shm is None:
            return None
        try:
            header = bytes(self.shm.buf[:self.header_size])
            fid, w, h = struct.unpack('QII', header)
            if fid == self.last_fid:
                return None
            self.last_fid = fid
            size = w * h * 3
            img_view = np.ndarray(
                (h, w, 3), dtype=np.uint8,
                buffer=self.shm.buf[self.header_size:self.header_size + size]
            )
            frame = img_view.copy()
            del img_view
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except (ValueError, struct.error, BufferError):
            return None

    def release(self):
        if self.shm:
            try:
                self.shm.close()
            except:
                pass


# ============ 主测试函数 ============
def run_http_test(url):
    """
    HTTP模式：从Windows远程连接板卡视频流
    做边缘检测 + 路径规划（不需要RKNN环境）
    """
    print("=" * 60)
    print("  视频流边缘检测 + 路径规划测试 (HTTP模式)")
    print("=" * 60)

    reader = HTTPStreamReader(url)
    edge_detector = EdgeDetector()
    planner = PathPlanner()

    # FPS统计
    fps_time = time.time()
    fps_count = 0
    cur_fps = 0.0

    print("\n按 ESC 退出\n")

    while True:
        frame = reader.read()
        if frame is None:
            if cv2.waitKey(1) == 27:
                break
            continue

        # 边缘检测
        edge_vis, edges, lines = edge_detector.detect(frame)

        # 路径规划（只用边缘检测，无分割mask）
        offset, center, info = planner.plan(None, lines, frame.shape)

        # 在边缘检测结果上画路径中心
        cx, cy = center
        cv2.circle(edge_vis, (cx, cy), 8, (0, 255, 0), -1)
        cv2.line(edge_vis, (cx, 0), (cx, frame.shape[0]), (0, 255, 0), 2)

        # 显示偏移量
        direction = "LEFT" if offset < -0.05 else "RIGHT" if offset > 0.05 else "STRAIGHT"
        cv2.putText(edge_vis, f"Offset: {offset:.3f} ({direction})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            cur_fps = fps_count / (time.time() - fps_time)
            fps_count = 0
            fps_time = time.time()
        cv2.putText(edge_vis, f"FPS: {cur_fps:.1f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示
        # 缩小显示（如果分辨率太大）
        display_h = 480
        scale = display_h / frame.shape[0]
        display = cv2.resize(edge_vis, None, fx=scale, fy=scale)
        cv2.imshow("Edge Detection + Path Planning", display)

        if cv2.waitKey(1) == 27:
            break

    reader.release()
    cv2.destroyAllWindows()
    print("测试结束")


def run_shm_test():
    """
    SHM模式：在板卡本地运行
    语义分割(NPU) + 边缘检测 + 路径规划
    """
    print("=" * 60)
    print("  语义分割 + 边缘检测 + 路径规划 (SHM模式)")
    print("=" * 60)

    # 导入RKNN相关模块（仅在板卡上可用）
    from seg_infer import SegWrap
    from infer_wrap import InferWrap

    reader = SHMVideoReader()
    if not reader.connect():
        return

    # 加载模型
    print("\n[1/2] 加载语义分割模型...")
    seg = SegWrap(model_dir="road/model", TPEs=1)

    print("[2/2] 加载目标检测模型...")
    detector = InferWrap(model_dir="model", TPEs=2)

    edge_detector = EdgeDetector()
    planner = PathPlanner()

    # 预热
    print("\n预热模型...")
    test_frame = reader.read()
    while test_frame is None:
        test_frame = reader.read()
        time.sleep(0.01)

    seg_result, _ = seg.infer(test_frame)
    det_result, _ = detector.infer(test_frame)
    print("预热完成!\n")

    # FPS统计
    fps_time = time.time()
    fps_count = 0
    cur_fps = 0.0

    print("按 ESC 退出\n")

    while True:
        frame = reader.read()
        if frame is None:
            if cv2.waitKey(1) == 27:
                break
            continue

        # 1. 语义分割
        seg_result, seg_flag = seg.infer(frame)
        if seg_result is not None:
            seg_frame, seg_mask, seg_offset = seg_result
        else:
            seg_frame = frame.copy()
            seg_mask = None
            seg_offset = 0.0

        # 2. 目标检测
        det_result, det_flag = detector.infer(frame)
        if det_result is not None:
            det_frame, boxes, classes, scores = det_result
        else:
            det_frame = frame.copy()
            boxes, classes, scores = None, None, None

        # 3. 边缘检测
        edge_vis, edges, lines = edge_detector.detect(frame)

        # 4. 路径规划（融合分割+边缘）
        offset, center, info = planner.plan(seg_mask, lines, frame.shape)

        # 5. 在分割结果上叠加路径规划信息
        cx, cy = center
        cv2.circle(seg_frame, (cx, cy), 8, (0, 0, 255), -1)
        cv2.line(seg_frame, (cx, 0), (cx, frame.shape[0]), (0, 0, 255), 2)

        direction = "LEFT" if offset < -0.05 else "RIGHT" if offset > 0.05 else "STRAIGHT"
        cv2.putText(seg_frame, f"PLAN: {direction} (offset={offset:.3f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(seg_frame, f"  seg={info['seg_offset']:.3f} edge={info['edge_offset']:.3f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            cur_fps = fps_count / (time.time() - fps_time)
            fps_count = 0
            fps_time = time.time()

        # 6. 拼接三路结果：检测 | 分割+路径 | 边缘
        h, w = frame.shape[:2]
        scale = 480 / h
        det_small = cv2.resize(det_frame, None, fx=scale, fy=scale)
        seg_small = cv2.resize(seg_frame, None, fx=scale, fy=scale)
        edge_small = cv2.resize(edge_vis, None, fx=scale, fy=scale)

        combined = np.hstack([det_small, seg_small, edge_small])
        cv2.putText(combined, f"FPS: {cur_fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, "Detection", (5, int(h*scale) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined, "Segmentation+Planning", (int(w*scale) + 5, int(h*scale) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined, "Edge Detection", (int(w*scale*2) + 5, int(h*scale) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Detection | Seg+Plan | Edge", combined)

        if cv2.waitKey(1) == 27:
            break

    # 清理
    seg.release()
    detector.release()
    reader.release()
    cv2.destroyAllWindows()
    print("测试结束")


# ============ 入口 ============
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='视频流语义分割+边缘检测测试')
    parser.add_argument('--mode', choices=['http', 'shm'], default='http',
                        help='http=远程HTTP流(Windows调试), shm=本地共享内存(板卡运行)')
    parser.add_argument('--url', default='http://192.168.1.85:8080/ar_feed',
                        help='HTTP视频流地址 (mode=http时使用)')
    args = parser.parse_args()

    if args.mode == 'http':
        run_http_test(args.url)
    else:
        run_shm_test()
