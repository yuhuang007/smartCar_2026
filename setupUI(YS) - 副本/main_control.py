"""
智能车主控程序
整合: 目标检测 + 语义分割 + OCR路牌识别 + 串口控制

比赛任务对应:
  Task 1 安全避让: 检测行人 → 制动/绕行 (第1圈)
  Task 2 语义导航: OCR路牌 → 文心大模型 → 左右转 (第2圈)
  Task 4 全域巡航: 语义分割巡线 + 金币收集 (全3圈)

运行环境: X-01板卡(RK3588) 上的 setupUI/dist/dlis/ 目录
视频来源: X-Verse AR引擎的共享内存(SHM)

使用方式:
  cd ~/Desktop/setupUI/dist/dlis/
  python3 main_control.py
"""

import time
import struct
import numpy as np
import cv2
from multiprocessing import shared_memory, resource_tracker

# 导入推理库
from infer_wrap import InferWrap          # 目标检测
from seg_infer import SegWrap              # 语义分割
from serial_comm.serial_control import SerialController, UART1Controller  # 串口控制(二进制帧+ASCII指令)
from stream_seg_test import EdgeDetector, PathPlanner, ArrowNavigator     # 边缘检测+路径规划+蓝色箭头导航


# ============ 全局配置 ============
CONFIG = {
    # 共享内存名称（与X-Verse引擎约定）
    "shm_name": "shm_ar_video",
    "shm_header_size": 16,

    # NPU核心分配
    "det_tpes": 2,      # 目标检测用2个NPU核心
    "seg_tpes": 1,      # 语义分割用1个NPU核心

    # 检测阈值
    "person_conf_thresh": 0.5,    # 行人检测置信度
    "coin_thresh": 0.3,           # 金币检测置信度

    # ===== 官方COCO 80类模型类别ID =====
    # person=0, car=2, traffic light=9, stop sign=13, sports ball=32
    "class_human": 0,             # 行人 (COCO: person)
    "class_car": 2,               # 小车 (COCO: car)
    "class_gold": 32,             # 金币 (COCO: sports ball 作为替代)
    "class_road_sign": 13,        # 路牌 (COCO: stop sign 作为替代)

    # 行人/障碍避让参数
    "person_safe_distance_y": 300,  # 行人在图像中y坐标小于此值=太近，需制动
    "person_safe_distance_x": 100,  # 行人在x方向偏移超过此值=可绕行

    # ===== 避让 & 金币吸引（控制虚拟车转向的偏移叠加）=====
    # 策略: 障碍挡在正前方→优先停车等它走(不出赛道)；侧方障碍→忽略
    "avoid_gain": 0.5,           # 避让转向强度(再加大往左靠防蹭侧身)
    "avoid_trigger_y": 0.2,      # 障碍框底超过此比例就避让(更早触发)
    "avoid_stop_y": 0.42,         # 障碍框底超过此比例=太近→停车(更早停防撞车头)
    "avoid_center_range": 0.28,   # 障碍中心偏移超此=在侧方(仅决定要不要轻微转向，不再一票否决)
    "avoid_bigbox_h": 0.45,       # 障碍框高超画面此比例=怗脸了，无论在哪都停车
    "avoid_min_box_h": 0.14,      # 障碍框高占画面不到此比例=太远的小目标，忽略
    "coin_gain": 0.25,           # 金币吸引转向强度
    "coin_road_halfwidth": 0.45,  # 道路半宽（归一化），金币横向超出此范围=道路外不吃
    "coin_min_y": 0.3,            # 金币中心y小于此比例=太远/天空，不追

    # ===== 蓝色箭头导航（本赛道主导航，地面蓝色箭头）=====
    # HSV阈值: 如果箭头没被正确提取，看调试窗口的蓝色mask覆盖情况微调
    "arrow_hsv_lower": (85, 50, 80),    # 蓝/青色下限 (H,S,V)
    "arrow_hsv_upper": (125, 255, 255), # 蓝/青色上限
    "arrow_near_weight": 0.6,     # 近带权重（越大越跟当前位置，越小越看前方）
    "arrow_steer_gain": 1.0,      # 箭头导航转向增益（转不够猛调大）
    # ★赛道丢失恢复：蓝色赛道跑出视野时，不能直走出图！
    # 而是朝“最后一次看到赛道的方向”强力打方向，把赛道转回视野
    "arrow_lost_recover_gain": 0.4,  # 丢失时的固定转向强度（越大转回来越快，太大会转过头）
    "arrow_lost_speed": 0.5,         # 丢失时的速度比例（减速给转向找回赛道的时间）0=原地转
    "arrow_lost_default_dir": -0.1,  # 正前方丢失(无历史方向)时默认搜索方向：<0左转 >0右转


    # 巡线控制参数 (PID-like 简化版)
    "lane_center_offset_threshold": 0.05,  # 偏移阈值
    "steering_center": 1500,               # 舵机居中值
    "steering_left": 1200,                 # 左转值
    "steering_right": 1800,                # 右转值
    "steering_sensitivity": 3000,          # 转向灵敏度(偏移量×灵敏度=转向调整量)

    # 速度参数
    "normal_speed": 500,
    "slow_speed": 300,
    "stop_speed": 0,

    # 当前圈数（用于任务调度）
    "current_lap": 1,

    # 串口配置
    "serial_port": "/dev/ttyUSB0",
    "serial_baudrate": 115200,

    # 调试模式（不连下位机，仅打印指令）
    "debug_mode": True,

    # ===== 虚拟车体控制（控制画面中的虚拟车，类似网页WASD）=====
    "enable_virtual_car": True,   # 是否启用虚拟车体控制
    "vcar_ip": "127.0.0.1",       # AR引擎地址（板卡本地）
    "vcar_port": 9005,            # 控制端口（对应control_port）
    "vcar_max_speed": 0.05,       # 最大速度（与网页速度滑块一致）
    "vcar_turn_rate": 5.0,        # 转向速率（度/步）
    # 启动对准: 原地转到赛道在正前方再出发(防开局横穿地图)
    "vcar_startup_turn_deg": 20.0,   # (已改为自适应对准模式，此值仅保留兼容)
    "vcar_startup_turn_step": 2.0,   # 启动转向每帧转多少度（越小越平滑）
    "startup_align_threshold": 0.1,  # 起步对准: 赛道偏移<此值=已对准，出发
    "startup_max_turn": 360.0,       # 起步最多原地转一圈，兜底防卡死
    # 开局航点导航: 先跑到此坐标点再切换赛道导航(不改变初始起点0,0,0)
    "waypoint_start": [0.63, 0.0, 1.17],  # 开局第一个目标点[x,y,z]，None=不走航点直接对准
    "waypoint_arrive_dist": 0.15,         # 距目标小于此值=到达，切换赛道导航
    # 初始位姿 [x, y, z, pitch, yaw, roll]（绝对坐标）
    # ★重要: 用网页WASD把车开到赛道起点，记下网页显示的 X/Y/Z/YAW 填到这里
    # 否则车会从(0,0,0)原点开始，可能不在赛道上。None=从原点开始
    "vcar_init_pose": None,       # 例: [0.39, 0.0, 1.27, 0.0, 22.0, 0.0]

    # 是否启用OCR（需要安装PaddleOCR；未安装时自动降级为黄色路牌检测）
    "enable_ocr": True,
    # 路牌转向（识别到指示牌→右转/左转，接入虚拟车 final_offset）
    "sign_turn_gain": 0.25,        # 路牌转向强度(降低防冲出赛道)
    "sign_hold_frames": 30,        # 持续偏移帧数(30帧≈37度够进含路)
    "sign_ocr_interval": 3.0,      # 大模型调用慢, 3秒一次防卡顿
    "ernie_access_token": "",  # 百度AI Studio access_token(erniebot SDK用)
    "qianfan_api_key": "",  # 千帆v2 API Key(调ernie-5.0)

    # ===== UART1 真实小车控制(ASCII指令协议) =====
    "uart1_enable": True,        # 是否启用UART1真实小车同步控制
    "uart1_port": "/dev/ttyUSB0",  # USB转串口设备路径(蓝色USB口)
    "uart1_baudrate": 115200,    # 波特率(文档要求115200)
    "uart1_max_duty": 10,        # 最大电机占空比%(4×5m小地图, 建议8~12)
    "uart1_servo_center": 90,    # 舵机中位角度(90=居中)
    "uart1_servo_range": 45,     # 舵机偏移范围(±45度, 即45~135)
    "uart1_servo_invert": True, # 舵机方向反转(转向反了设True)

    # 是否显示可视化窗口
    "show_window": True,
}


# ============ 共享内存读取器 ============
class ARVideoReader:
    """
    从X-Verse引擎的共享内存读取AR叠加后的视频帧

    与 ar_receiver.py 中的逻辑相同：
    - 头部16字节: 帧号(8B Q) + 宽(4B I) + 高(4B I)
    - 后面是RGB像素数据
    """

    def __init__(self, shm_name="shm_ar_video", header_size=16):
        self.shm_name = shm_name
        self.header_size = header_size
        self.shm = None
        self.last_fid = 0

    def connect(self, max_wait=30):
        """连接共享内存"""
        print("[ARReader] 等待X-Verse引擎启动...")
        wait_count = 0
        while wait_count < max_wait:
            try:
                self.shm = shared_memory.SharedMemory(name=self.shm_name)
                # 防止客户端退出时误删服务器的SHM
                try:
                    resource_tracker.unregister('/' + self.shm_name, 'shared_memory')
                except:
                    pass
                print("[ARReader] 已连接共享内存!")
                return True
            except FileNotFoundError:
                time.sleep(1.0)
                wait_count += 1
                print(f"[ARReader] 等待中... ({wait_count}/{max_wait})")
        print("[ARReader] 连接超时!")
        return False

    def read_frame(self):
        """
        读取一帧

        返回:
            frame: numpy.ndarray (BGR) 或 None
        """
        if self.shm is None:
            return None

        try:
            header = bytes(self.shm.buf[:self.header_size])
            fid, w, h = struct.unpack('QII', header)

            if fid == self.last_fid:
                return None  # 帧没变

            self.last_fid = fid

            size = w * h * 3
            img_view = np.ndarray(
                (h, w, 3),
                dtype=np.uint8,
                buffer=self.shm.buf[self.header_size: self.header_size + size]
            )
            frame = img_view.copy()
            del img_view

            # 翻转+颜色转换（与ar_receiver.py一致）
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            return frame

        except (ValueError, struct.error, BufferError):
            return None

    def close(self):
        if self.shm:
            try:
                self.shm.close()
            except:
                pass


# ============ 主控程序 ============
class SmartCarController:
    """
    智能车主控制器

    整合目标检测、语义分割、OCR和串口控制
    """

    def __init__(self, config=CONFIG):
        self.config = config

        # 初始化各模块
        print("=" * 60)
        print("  智能车主控程序启动")
        print("=" * 60)

        # 1. 目标检测
        print("\n[1/5] 加载目标检测模型...")
        self.detector = None
        try:
            self.detector = InferWrap(model_dir="model", TPEs=config["det_tpes"])
        except Exception as e:
            print(f"[WARN] 目标检测模型加载失败，进入无检测模式:")
            print(f"       {e}")
            print(f"       检查: 1.模型文件是否损坏 2.是否用二进制模式传输")

        # 2. 语义分割
        print("[2/5] 加载语义分割模型...")
        self.seg = None
        try:
            self.seg = SegWrap(model_dir="road/model", TPEs=config["seg_tpes"])
        except Exception as e:
            print(f"[WARN] 语义分割模型加载失败，进入无分割模式:")
            print(f"       {e}")

        # 3. OCR（可选）
        if config["enable_ocr"]:
            print("[3/5] 加载OCR引擎...")
            from ocr_sign.ocr_engine import OCRSignRecognizer
            self.ocr = OCRSignRecognizer(
                ernie_access_token=config.get("ernie_access_token", ""),
                qianfan_api_key=config.get("qianfan_api_key", "")
            )
        else:
            print("[3/5] OCR未启用 (enable_ocr=False)")
            self.ocr = None

        # 4. 串口控制
        print("[4/5] 初始化串口通信...")
        if config["debug_mode"]:
            self.serial = SerialController(port="MOCK_PORT")
        else:
            self.serial = SerialController(
                port=config["serial_port"],
                baudrate=config["serial_baudrate"]
            )

        # 5. 边缘检测 + 路径规划 + 蓝色箭头导航
        print("[5/5] 初始化边缘检测与路径规划...")
        self.edge_detector = EdgeDetector()
        self.path_planner = PathPlanner()
        # 蓝色箭头导航（本赛道主导航）: HSV阈值可在CONFIG调
        self.arrow_nav = ArrowNavigator(
            lower_hsv=tuple(config.get("arrow_hsv_lower", (85, 50, 80))),
            upper_hsv=tuple(config.get("arrow_hsv_upper", (125, 255, 255))),
            near_weight=config.get("arrow_near_weight", 0.6),
        )

        # 6. 虚拟车体控制（控制画面中的虚拟车，类似网页WASD）
        self.vcar = None
        if config.get("enable_virtual_car", False):
            print("[6] 初始化虚拟车体控制...")
            try:
                from virtual_car import VirtualCarController
                # 解析初始位姿配置 [x,y,z,pitch,yaw,roll] → dict
                init_pose = None
                ip = config.get("vcar_init_pose", None)
                if ip is not None and len(ip) == 6:
                    init_pose = {"x": ip[0], "y": ip[1], "z": ip[2],
                                 "pitch": ip[3], "yaw": ip[4], "roll": ip[5]}
                self.vcar = VirtualCarController(
                    target_ip=config["vcar_ip"],
                    target_port=config["vcar_port"],
                    init_pose=init_pose,
                )
                self.vcar.max_speed = config["vcar_max_speed"]
                self.vcar.turn_rate = config["vcar_turn_rate"]
            except Exception as e:
                print(f"[WARN] 虚拟车体控制初始化失败: {e}")
                self.vcar = None

        # 7. UART1 真实小车同步控制(与虚拟车并行)
        self.uart1 = None
        if config.get("uart1_enable", False):
            print("[7] 初始化UART1真实小车控制...")
            try:
                self.uart1 = UART1Controller(
                    port=config["uart1_port"],
                    baudrate=config["uart1_baudrate"],
                    max_duty=config["uart1_max_duty"],
                    servo_center=config["uart1_servo_center"],
                    servo_range=config["uart1_servo_range"],
                    servo_invert=config.get("uart1_servo_invert", False),
                )
            except Exception as e:
                print(f"[WARN] UART1初始化失败: {e}")
                self.uart1 = None

        # AR视频读取器
        self.ar_reader = ARVideoReader(
            shm_name=config["shm_name"],
            header_size=config["shm_header_size"]
        )

        # FPS统计
        self.fps_time = time.time()
        self.fps_count = 0
        self.current_fps = 0.0

        # 运行状态
        self.running = False
        self.current_lap = config["current_lap"]

        # 初始化检测模型预热
        self._det_warmup = False
        self._seg_warmup = False

    def _warmup(self, frame):
        """预热模型"""
        if not self._det_warmup and self.detector is not None:
            try:
                self.detector.infer(frame)
                self._det_warmup = True
                print("[Warmup] 目标检测模型预热完成")
            except Exception as e:
                print(f"[WARN] 目标检测预热失败: {e}")
                self._det_warmup = True  # 标记已尝试，不再重试

        if not self._seg_warmup and self.seg is not None:
            try:
                self.seg.infer(frame)
                self._seg_warmup = True
                print("[Warmup] 语义分割模型预热完成")
            except Exception as e:
                print(f"[WARN] 语义分割预热失败: {e}")
                self._seg_warmup = True

    # ========== Task 1: 行人避让 ==========
    def task_avoid_person(self, det_frame, boxes, classes, scores):
        """
        行人/汽车避让决策（返回避让偏移量 + 速度比例）

        核心策略（障碍都在赛道上，不能猛打方向出赛道）:
        1. 只处理挡在正前方、够近、够大的障碍（过滤远处/侧方目标）
        2. 障碍很近 → 停车等它走（speed=0，不打方向，保持在赛道上）
        3. 障碍中距离 → 减速 + 轻微避让
        4. 障碍在侧方（没挡路）→ 忽略，继续沿赛道走

        返回:
            avoid_offset: 避让转向偏移 (-0.5~+0.5)，0=无障碍/停车
            speed_ratio:  速度比例 (0=停, 0.4=减速, 1.0=正常)
        """
        h, w = det_frame.shape[:2]
        avoid_offset = 0.0
        speed_ratio = 1.0
        nearest_bottom = -1   # 最近障碍的框底y（用于选最近）
        nearest_info = None

        if boxes is not None:
            for box, cl, score in zip(boxes, classes, scores):
                is_person = (cl == self.config["class_human"] and
                             score > self.config["person_conf_thresh"])
                is_car = (cl == self.config["class_car"] and score > 0.4)
                if not (is_person or is_car):
                    continue

                left, top, right, bottom = [int(b) for b in box]
                obj_cx = (left + right) / 2
                bottom_ratio = bottom / h        # 框底占高比，越大越近
                box_h_ratio = (bottom - top) / h # 框高占画面比，越大越近
                obj_offset = (obj_cx / w) - 0.5  # 障碍偏移: <0在左, >0在右

                # 过滤一: 远处的小目标（框太小）不管
                if box_h_ratio < self.config["avoid_min_box_h"]:
                    continue
                # 过滤二: 框底太靠上（远）不管
                if bottom_ratio < self.config["avoid_trigger_y"]:
                    continue
                # ★ 不再用 center_range 一票否决！怗脸的大障碍可能中心偏左/右，
                #    不能因为偏一点就当侧方忽略（之前撞车的元凶）

                # 画框
                color = (0, 0, 255) if is_person else (255, 128, 0)
                label = "PERSON" if is_person else "CAR"
                cv2.rectangle(det_frame, (left, top), (right, bottom), color, 3)
                cv2.putText(det_frame, f"{label} {score:.2f}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 只处理最近的障碍
                if bottom <= nearest_bottom:
                    continue
                nearest_bottom = bottom
                nearest_info = (bottom_ratio, obj_offset, box_h_ratio)

        # 对最近的障碍做决策
        if nearest_info is not None:
            bottom_ratio, obj_offset, box_h_ratio = nearest_info
            # ① 怗脸/够近 → 无条件停车（无论障碍在画面哪个位置）
            if (bottom_ratio >= self.config["avoid_stop_y"] or
                    box_h_ratio >= self.config["avoid_bigbox_h"]):
                speed_ratio = 0.0
                avoid_offset = 0.0
            # ② 中距离且挡在正前方 → 减速 + 轻微避让
            elif abs(obj_offset) <= self.config["avoid_center_range"]:
                speed_ratio = 0.15
                proximity = bottom_ratio / self.config["avoid_stop_y"]
                # 正前方障碍: sign(0)=0不避让会直撞, 强制往左偏
                if abs(obj_offset) < 0.05:
                    avoid_offset = -self.config["avoid_gain"] * proximity
                else:
                    avoid_offset = -np.sign(obj_offset) * self.config["avoid_gain"] * proximity
            # ③ 中距离在侧方 → 减速警惕，不强行转向（避免出赛道）
            else:
                speed_ratio = 0.6
                avoid_offset = 0.0

            # 可视化提示
            action = "STOP" if speed_ratio <= 0.0 else \
                     ("AVOID R" if avoid_offset > 0 else "AVOID L")
            cv2.putText(det_frame, f"ACTION: {action}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if speed_ratio <= 0.0:
                self.serial.brake()

        return avoid_offset, speed_ratio

    # ========== Task 4: 巡线控制 ==========
    def task_lane_following(self, seg_frame, plan_offset, plan_info=None):
        """
        语义分割 + 边缘检测融合巡线控制

        逻辑:
        1. PathPlanner 已融合分割mask + 边缘车道线，输出平滑后的偏移量
        2. 根据偏移量计算转向角度
        3. 偏移量大→大角度转向, 偏移量小→微调

        参数:
            seg_frame: 分割可视化帧（用于叠加信息）
            plan_offset: 路径规划器输出的融合偏移量 (-0.5~+0.5)
            plan_info: dict, 路径规划详细信息（seg_offset, edge_offset等）
        """
        # 计算转向角度
        # plan_offset范围: -0.5(最左) ~ +0.5(最右)
        # steering范围: 500(最左) ~ 2500(最右), 1500=居中
        steering = int(
            self.config["steering_center"] +
            plan_offset * self.config["steering_sensitivity"]
        )
        steering = max(500, min(2500, steering))  # 限幅

        # 发送转向指令
        self.serial.set_steering(steering)

        # 如果没有检测到障碍物，保持正常速度
        self.serial.set_speed(self.config["normal_speed"])

        # 可视化
        direction = "LEFT" if plan_offset < -0.05 else \
                    "RIGHT" if plan_offset > 0.05 else "STRAIGHT"
        cv2.putText(seg_frame, f"PLAN: {direction} (offset={plan_offset:.3f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 显示分割和边缘各自的偏移量
        if plan_info:
            cv2.putText(seg_frame,
                        f"  seg={plan_info['seg_offset']:.3f} edge={plan_info['edge_offset']:.3f}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return steering

    # ========== Task 2: 路牌OCR（检测到路牌→停车OCR→转向）==========
    def task_sign_recognition(self, det_frame, boxes, classes, scores):
        """
        路牌OCR识别 → 返回导航方向
        每0.5秒用OpenCV检测路牌区域(快)，只在检测到路牌时才调用OCR(慢，阻塞几秒)
        调用OCR后5秒冷却期，防止小车在路牌附近循环检测
        """
        if not hasattr(self, "_sign_direction"):
            self._sign_direction = "straight"
            self._sign_hold = 0
            self._sign_last_ocr = 0.0  # 上次调用OCR的时间
            self._sign_last_detect = 0.0  # 上次检测路牌区域的时间

        h, w = det_frame.shape[:2]
        now = time.time()

        # 保持计数：识别到方向后持续转向 hold 帧
        if self._sign_hold > 0:
            self._sign_hold -= 1
        else:
            self._sign_direction = "straight"

        # 每0.5秒检测路牌区域 + OCR调用后10秒冷却(防循环)
        if (self._sign_hold <= 0
                and now - self._sign_last_detect >= 0.5
                and now - self._sign_last_ocr >= 10.0):
            self._sign_last_detect = now
            top_third = det_frame[0:int(h * 0.40), :]
            sign_img = self._detect_sign_by_color(top_third)

            if sign_img is not None:
                # 检测到路牌 → 停车调用OCR(阻塞几秒)
                self._sign_last_ocr = now  # 更新OCR时间, 10秒冷却
                print(f"[Sign] 检测到路牌 → 停车调用OCR")
                if self.uart1: self.uart1.send_stop()  # ★OCR前停车
                if self.ocr is not None:
                    new_dir, raw = self.ocr.ask_ernie_with_image(sign_img)
                    print(f"[Sign] OCR结果 → {new_dir} ({raw})")
                    cv2.putText(det_frame, f"SIGN:{new_dir}",
                                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    # 不管什么方向都设hold, 让小车驶过路牌(防循环检测)
                    self._sign_direction = new_dir
                    self._sign_hold = self.config.get("sign_hold_frames", 60)
                else:
                    self._sign_direction = "right"
                    self._sign_hold = self.config.get("sign_hold_frames", 60)
                    print(f"[Sign] OCR不可用 → 默认right")

        cv2.putText(det_frame, f"SIGN:{self._sign_direction} h{self._sign_hold}",
                    (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        return True, self._sign_direction

    def _detect_sign_by_color(self, top_img):
        """
        OpenCV蓝底白字路牌检测(不依赖paddlepaddle)
        原理: 路牌=蓝底白字，蓝块内有高亮度像素(白字)。
              拱门=纯蓝无白字; 赛道箭头=纯蓝发光。用白色像素比例区分。
        返回: 路牌区域图片(np.ndarray) 或 None
        """
        hsv = cv2.cvtColor(top_img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, (85, 30, 30), (130, 255, 255))
        total = top_img.shape[0] * top_img.shape[1]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask)
        best = (0.0, 0.0, 0.0)  # (蓝色面积比, 亮度std, 白字比例)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            blue_ratio = area / total
            if blue_ratio < 0.01:  # 最小1%
                continue
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            roi_gray = cv2.cvtColor(top_img[by:by+bh, bx:bx+bw], cv2.COLOR_BGR2GRAY)
            std = float(np.std(roi_gray))
            white_ratio = float(np.sum(roi_gray > 160)) / roi_gray.size
            if blue_ratio > best[0]:
                best = (blue_ratio, std, white_ratio)
            # 蓝块>2% + std>40 + 白字>10% → 路牌(近了才触发, 避免远处文字不全)
            if blue_ratio >= 0.02 and std > 40 and white_ratio > 0.10:
                print(f"[Sign] 蓝块{blue_ratio:.1%} std{std:.0f} 白{white_ratio:.0%} → 路牌！")
                return top_img[by:by+bh, bx:bx+bw].copy()
        if best[0] > 0.01:
            print(f"[Sign] max蓝{best[0]:.1%} std{best[1]:.0f} 白{best[2]:.0%} → 非路牌")
        return None

    # ========== 金币检测 ==========
    def detect_coins(self, det_frame, boxes, classes, scores):
        """
        金币检测 + 吸引（贯穿全程，每枚-5s奖励）

        金币类别: class_gold
        策略: 选最近的【道路内】金币，输出一个吸引偏移，让车往金币方向碰
              金币在道路外（横向偏移过大 / 太远）则不追

        返回:
            coin_offset: 金币吸引偏移 (-0.5~+0.5)，0=无可吃金币
        """
        h, w = det_frame.shape[:2]
        coin_count = 0
        coin_offset = 0.0
        target_bottom = -1   # 选最近金币

        if boxes is not None:
            for box, cl, score in zip(boxes, classes, scores):
                if cl != self.config["class_gold"] or score <= self.config["coin_thresh"]:
                    continue
                coin_count += 1
                left, top, right, bottom = [int(b) for b in box]
                coin_cx = (left + right) / 2
                coin_cy = (top + bottom) / 2
                cx_norm = (coin_cx / w) - 0.5    # 横向偏移
                cy_ratio = coin_cy / h

                # 画框
                cv2.rectangle(det_frame, (left, top), (right, bottom),
                              (0, 255, 255), 2)
                cv2.putText(det_frame, f"COIN {score:.2f}", (left, top - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # 道路内判定: 横向不超道路半宽 + 在前方地面（不在天空）
                in_road = (abs(cx_norm) <= self.config["coin_road_halfwidth"] and
                           cy_ratio >= self.config["coin_min_y"])
                if not in_road:
                    continue

                # 选最近（框底最靠下）的道路内金币
                if bottom <= target_bottom:
                    continue
                target_bottom = bottom
                # 吸引: 往金币方向偏（cx_norm>0在右→右偏）
                coin_offset = cx_norm * self.config["coin_gain"] * 2.0
                coin_offset = max(-0.5, min(0.5, coin_offset))
                # 高亮目标金币
                cv2.rectangle(det_frame, (left, top), (right, bottom),
                              (0, 200, 0), 3)

        if coin_count > 0:
            cv2.putText(det_frame, f"COINS: {coin_count}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return coin_offset

    # ========== 主循环 ==========
    def run(self):
        """主循环"""
        # 连接AR视频流
        if not self.ar_reader.connect():
            print("[ERROR] 无法连接AR视频流，请确保X-Verse引擎已启动")
            return

        print("\n" + "=" * 60)
        print("  系统就绪，开始自动驾驶!")
        print(f"  当前圈数: {self.current_lap}")
        print(f"  调试模式: {'是' if self.config['debug_mode'] else '否'}")
        print("=" * 60 + "\n")

        self.running = True
        self._run_start_time = time.time()  # 记录启动时间, 用于开局保护期

        # 启动对准阶段状态: 原地转到赛道在正前方再出发(防止开局横穿地图)
        self._startup_done = False
        self._waypoint_done = False
        self._startup_turned = 0.0

        while self.running:
            try:
                # 1. 读取AR视频帧
                frame = self.ar_reader.read_frame()
                if frame is None:
                    if cv2.waitKey(1) == 27:
                        break
                    continue

                # 2. 预热模型（第一帧时）
                self._warmup(frame)

                # 3. 目标检测推理
                # infer() 返回 (result, flag)
                # result = (画框图像, boxes, classes, scores)
                if self.detector is not None:
                    det_result, det_flag = self.detector.infer(frame)
                    if det_result is not None:
                        det_frame, boxes, classes, scores = det_result
                    else:
                        det_frame = frame.copy()
                        boxes, classes, scores = None, None, None
                else:
                    det_frame = frame.copy()
                    boxes, classes, scores = None, None, None

                # 4. 语义分割推理
                # infer() 返回 (result, flag)
                # result = (可视化图像, mask, lane_offset)
                if self.seg is not None:
                    seg_result, seg_flag = self.seg.infer(frame)
                    if seg_result is not None:
                        seg_frame, seg_mask, lane_offset = seg_result
                    else:
                        seg_frame = frame.copy()
                        seg_mask = None
                        lane_offset = 0.0
                else:
                    seg_frame = frame.copy()
                    seg_mask = None
                    lane_offset = 0.0

                # 5. 获取检测结果
                # det_result已在上一步解包: det_frame, boxes, classes, scores

                # 6. 蓝色箭头导航（本赛道主导航）
                arrow_offset, arrow_vis, arrow_valid, arrow_info = \
                    self.arrow_nav.navigate(frame)

                # 7. 边缘检测（作为箭头丢失时的回退）
                edge_vis, edge_mask, edge_lines = self.edge_detector.detect(frame)

                # 8. 确定基础巡线偏移: 优先蓝色箭头
                if arrow_valid:
                    base_offset = arrow_offset * self.config.get("arrow_steer_gain", 1.0)
                    plan_info = {"seg_offset": 0.0, "edge_offset": arrow_offset,
                                 "is_junction": False}
                    speed_ratio = 1.0
                else:
                    # ★箭头丢失（蓝色赛道跑出视野）→ 绝不能直走出图！
                    #   朝“最后一次看到赛道的方向”强力打方向，把赛道转回视野
                    last = self.arrow_nav.last_offset
                    if abs(last) < 0.02:
                        # 正前方丢失(无明显方向) → 默认搜索方向
                        last = self.config.get("arrow_lost_default_dir", -0.1)
                    recover = self.config.get("arrow_lost_recover_gain", 0.4)
                    base_offset = np.sign(last) * recover
                    plan_info = {"seg_offset": 0.0, "edge_offset": base_offset,
                                 "is_junction": True}
                    # 丢失时减速，给转向找回赛道的时间
                    speed_ratio = self.config.get("arrow_lost_speed", 0.5)
                base_offset = max(-0.5, min(0.5, base_offset))

                # 8.5 base_offset低通滤波(抑制路口质心横摆)
                if not hasattr(self, '_smooth_base'):
                    self._smooth_base = 0.0
                _alpha = self.config.get("base_smooth_alpha", 1.0)  # 1.0=不滤波(只靠ArrowNavigator内部平滑)
                base_offset = _alpha * base_offset + (1 - _alpha) * self._smooth_base
                self._smooth_base = base_offset

                # 9. 行人/汽车避让（全程）→ 避让偏移 + 速度
                avoid_offset, avoid_speed = self.task_avoid_person(
                    det_frame, boxes, classes, scores)
                speed_ratio = min(speed_ratio, avoid_speed)

                # 10. 金币吸引（全程）→ 仅在无障碍避让时去吃道路内金币
                coin_offset = 0.0
                if abs(avoid_offset) < 0.01:
                    coin_offset = self.detect_coins(
                        det_frame, boxes, classes, scores)

                # 11. 路牌OCR → 每帧检测路牌区域, 检测到才调OCR(不盲目定期调用)
                _, sign_dir = self.task_sign_recognition(det_frame, boxes, classes, scores)
                sign_offset = 0.0
                if sign_dir == "right":
                    sign_offset = +self.config.get("sign_turn_gain", 0.12)
                elif sign_dir == "left":
                    sign_offset = -self.config.get("sign_turn_gain", 0.12)

                # 12.5 决策融合: 路牌转向优先（覆盖巡线/金币）；避让仍可叠加
                if abs(sign_offset) > 0.01:
                    final_offset = sign_offset + avoid_offset  # 路牌轻偏(不叠加base, 避免被巡线抵消)
                    speed_ratio = max(speed_ratio, 0.5)   # 路牌转向时保证车动着转
                elif abs(avoid_offset) > 0.05:
                    final_offset = avoid_offset + base_offset * 0.3  # 避让为主, 巡线只占30%防抵消
                else:
                    final_offset = base_offset + avoid_offset + coin_offset
                final_offset = max(-0.5, min(0.5, final_offset))

                # 12.7 横摆抑制 + 路口靠左
                if not hasattr(self, '_prev_final'):
                    self._prev_final = 0.0
                    self._prev_arrow = 0.0
                    self._intersection_hold = 0
                # 路口检测: arrow_offset符号反转 = 质心横摆 = 到路口了
                if arrow_valid and self._prev_arrow * arrow_offset < -0.01:
                    self._intersection_hold = self.config.get("intersection_hold_frames", 20)
                    print(f"[VCar] 路口(arrow {self._prev_arrow:+.2f}→{arrow_offset:+.2f}) 靠左{self._intersection_hold}帧")
                self._prev_arrow = arrow_offset
                # 路口且没路牌 → 默认靠左(防横摆冲出赛道)
                if self._intersection_hold > 0 and abs(sign_offset) < 0.01:
                    final_offset = self.config.get("intersection_left_bias", -0.2)
                    self._intersection_hold -= 1
                # 变化率限制(仅路牌转向不受限, 路口靠左也受限防突变): 每帧offset变化≤max_delta
                if abs(sign_offset) < 0.01:
                    max_delta = self.config.get("offset_max_delta", 0.4)  # 0.4=基本不限(三重平滑会滞后冲出)
                    if abs(final_offset - self._prev_final) > max_delta:
                        final_offset = self._prev_final + np.sign(final_offset - self._prev_final) * max_delta
                self._prev_final = final_offset
                final_offset = max(-0.5, min(0.5, final_offset))

                # 可视化（用最终偏移，画到箭头导航图上）
                steering = self.task_lane_following(arrow_vis, final_offset, plan_info)

                # 12.6 驱动虚拟车体（final_offset控转向 + speed_ratio控速度）
                if self.vcar is not None:
                    yaw_before = self.vcar.pose["yaw"]   # 命令前的yaw
                    wp = self.config.get("waypoint_start", None)
                    if wp is not None and not self._waypoint_done:
                        # ① 航点导航: 朝目标点开，到达后切换赛道导航(不改变起点0,0,0)
                        drive_mode = "WAYPOINT"
                        arrive = self.config.get("waypoint_arrive_dist", 0.15)
                        p = self.vcar.pose
                        dx = wp[0] - p["x"]
                        dz = wp[2] - p["z"]
                        dist = float(np.sqrt(dx * dx + dz * dz))
                        if dist < arrive:
                            self._waypoint_done = True
                            self._startup_done = True
                            self._transition_hold = 10  # 过渡10帧减速, 等ArrowNavigator稳定
                            print(f"[VCar] 到达航点({wp[0]},{wp[2]}) dist={dist:.2f} 切换赛道导航(过渡减速)")
                        else:
                            # 朝目标点转向: 运动模型 x+=sin(yaw),z+=cos(yaw)，故目标角=atan2(dx,dz)
                            target_yaw = float(np.degrees(np.arctan2(dx, dz)))
                            yaw_err = (target_yaw - p["yaw"] + 180) % 360 - 180
                            offset = max(-0.5, min(0.5, yaw_err / 30.0))
                            self.vcar.drive(offset, speed_ratio=1.0)
                    elif not self._startup_done:
                        # ② 启动对准(未配航点时): 原地转到赛道在正前方再出发
                        drive_mode = "STARTUP"
                        align_thresh = self.config.get("startup_align_threshold", 0.1)
                        max_turn = self.config.get("startup_max_turn", 360.0)
                        step = self.config.get("vcar_startup_turn_step", 2.0)
                        aligned = arrow_valid and abs(arrow_offset) < align_thresh
                        timeout = abs(self._startup_turned) >= max_turn
                        if aligned or timeout:
                            self._startup_done = True
                            print(f"[VCar] STARTUP对准完成 off={arrow_offset:.3f} "
                                  f"turned={self._startup_turned:.0f}"
                                  + ("(兼容兜底出发)" if timeout else ""))
                        else:
                            # 原地转: 朝赛道方向转(看不到赛道则默认右转找)
                            if arrow_valid:
                                if arrow_offset > 0:
                                    self.vcar.turn_right(step); self._startup_turned += step
                                else:
                                    self.vcar.turn_left(step); self._startup_turned -= step
                            else:
                                self.vcar.turn_right(step); self._startup_turned += step
                    elif speed_ratio <= 0.0:
                        # ② 避障停车
                        drive_mode = "STOP"
                        self.vcar.stop()
                    else:
                        # ③ 正常导航（巡线+避让+金币融合）
                        drive_mode = "DRIVE"
                        # 到达航点后过渡减速(前10帧慢速, 等offset稳定防冲出)
                        if hasattr(self, '_transition_hold') and self._transition_hold > 0:
                            self._transition_hold -= 1
                            speed_ratio = min(speed_ratio, 0.4)
                            drive_mode = "DRIVE*"
                        self.vcar.drive(final_offset, speed_ratio=speed_ratio)

                    # ★诊断日志: 每帧打印决策→虚拟车的完整链路数据
                    #   看点: 1) base/avoid/coin/final 偏移是否随赛道变化
                    #         2) mode 是否卡在 STARTUP/STOP
                    #         3) yaw 是否真的被改变（cmd后 vs cmd前）
                    #         4) x/z 是否真的在移动
                    if self.config.get("debug_mode", False):
                        p = self.vcar.pose
                        dyaw = p["yaw"] - yaw_before
                        print(f"[VCar] {drive_mode:7s} "
                              f"base={base_offset:+.3f} avoid={avoid_offset:+.3f} "
                              f"coin={coin_offset:+.3f} sign={sign_offset:+.3f}({sign_dir[:1]}) "
                              f"final={final_offset:+.3f} "
                              f"spd={speed_ratio:.1f} | arrow_valid={arrow_valid} "
                              f"| yaw={p['yaw']:6.1f}(d{dyaw:+.1f}) "
                              f"x={p['x']:+.2f} z={p['z']:+.2f} seq={self.vcar.seq}")

                    # ★统一用虚拟小车的实际运动增量驱动真实小车(虚实同步)
                    if self.uart1:
                        _p = self.vcar.pose
                        if not hasattr(self, '_uart_prev_yaw'):
                            self._uart_prev_x = _p['x']
                            self._uart_prev_z = _p['z']
                            self._uart_prev_yaw = _p['yaw']
                        ddx = _p['x'] - self._uart_prev_x
                        ddz = _p['z'] - self._uart_prev_z
                        ddist = float(np.sqrt(ddx*ddx + ddz*ddz))
                        ddyaw = (_p['yaw'] - self._uart_prev_yaw + 180) % 360 - 180
                        # 速度: 停车=0, 启动对准=最小速度, 其他=实际speed_ratio
                        if drive_mode == "STOP":
                            self.uart1.send_stop()  # 停车直接发S
                        else:
                            if drive_mode == "STARTUP":
                                uart_spd = 0.3  # 启动对准给30%速度
                            else:
                                uart_spd = speed_ratio
                            self.uart1.send_pose_delta(ddyaw, ddist, uart_spd)
                        self._uart_prev_x = _p['x']
                        self._uart_prev_z = _p['z']
                        self._uart_prev_yaw = _p['yaw']

                # 13. FPS统计
                self.fps_count += 1
                if time.time() - self.fps_time >= 1.0:
                    self.current_fps = self.fps_count / (time.time() - self.fps_time)
                    self.fps_count = 0
                    self.fps_time = time.time()

                # 14. 显示结果
                if self.config["show_window"]:
                    # 拼接三路结果：检测 | 蓝色箭头导航 | 边缘
                    h, w = frame.shape[:2]
                    scale = 400 / h
                    det_s = cv2.resize(det_frame, None, fx=scale, fy=scale)
                    seg_s = cv2.resize(arrow_vis, None, fx=scale, fy=scale)
                    edge_s = cv2.resize(edge_vis, None, fx=scale, fy=scale)
                    combined = np.hstack([det_s, seg_s, edge_s])
                    cv2.putText(combined, f"FPS: {self.current_fps:.1f} | Lap: {self.current_lap}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(combined, "Detection", (5, int(h*scale)+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(combined, "Arrow Nav", (int(w*scale)+5, int(h*scale)+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(combined, "Edge", (int(w*scale*2)+5, int(h*scale)+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.imshow("SmartCar - Det | Arrow Nav | Edge", combined)

                if cv2.waitKey(1) == 27:  # ESC退出
                    break

            except KeyboardInterrupt:
                print("\n[INFO] 用户中断")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

        # 清理
        self.cleanup()

    def cleanup(self):
        """清理资源"""
        print("\n[INFO] 清理资源...")
        self.running = False

        # 停车
        self.serial.set_speed(0)
        self.serial.brake()
        if self.uart1: self.uart1.send_stop()  # ★先显式停车再清理

        # 释放资源
        if self.detector is not None:
            self.detector.release()
        if self.seg is not None:
            self.seg.release()
        if self.vcar is not None:
            self.vcar.close()
        self.ar_reader.close()
        self.serial.close()
        if self.uart1: self.uart1.close()

        cv2.destroyAllWindows()
        print("[INFO] 系统已停止")


# ============ 入口 ============
if __name__ == '__main__':
    controller = SmartCarController(config=CONFIG)
    controller.run()
