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
import threading
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
    "avoid_gain": 0.4,           # 避让转向强度(0.3不够靠边, 提到0.4)
    "avoid_invert": False,      # 避让方向(新逻辑已正确, 不需反转)
    "avoid_trigger_y": 0.15,     # 障碍框底超过此比例就避让(0.2太晚, 0.15提前)
    "avoid_stop_y": 0.42,         # 障碍框底超过此比例=太近→停车(更早停防撞车头)
    "avoid_center_range": 0.28,   # 障碍中心偏移超此=在侧方(仅决定要不要轻微转向，不再一票否决)
    "avoid_bigbox_h": 0.45,       # 障碍框高超画面此比例=怗脸了，无论在哪都停车
    "avoid_min_box_h": 0.14,      # 障碍框高占画面不到此比例=太远的小目标，忽略
    # ★car专属避让参数(比行人更早触发+更强避让, car是移动目标反应慢会撞)
    "car_trigger_y": 0.08,       # car触发阈值(比通用0.15更早, 移动目标需提前反应)
    "car_min_box_h": 0.08,       # car最小框高(比通用0.14小, 远处小车也能检测)
    "car_avoid_gain": 0.33,      # car避让转向强度(满偏约18°, 含速度自适应增益)
    "car_avoid_speed": 0.75,      # car避让减速(最低D6, 不低于最低前进速度)
    "person_stop_y": 0.55,      # 行人停车阈值(比car的avoid_stop_y=0.42大, 靠近行人再停)
    "coin_gain": 0.25,           # 金币吸引转向强度
    "coin_road_halfwidth": 0.45,  # 道路半宽（归一化），金币横向超出此范围=道路外不吃
    "coin_min_y": 0.3,            # 金币中心y小于此比例=太远/天空，不追

    # ===== 蓝色箭头导航（本赛道主导航，地面蓝色箭头）=====
    # HSV阈值: 如果箭头没被正确提取，看调试窗口的蓝色mask覆盖情况微调
    "arrow_hsv_lower": (85, 50, 80),    # 蓝/青色下限 (H,S,V)
    "arrow_hsv_upper": (125, 255, 255), # 蓝/青色上限
    "arrow_near_weight": 0.65,    # 近带权重(0.5远带稀释当前偏差, 0.65当前偏差占主导转向更准)
    "arrow_steer_gain": 1.0,      # 箭头导航转向增益(1.5直道摆, 降到1.0)
    "arrow_dead_zone": 0.08,     # 死区: abs(offset)<此值不转向(消除HSV噪声持续偏转)
    "arrow_roi_margin_x": 0.08,  # ★左右边缘剔除比例(排除场地边缘/光线反射的干扰蓝色)
    "arrow_min_pixels": 120,     # ★最小像素门槛(80零星干扰易误判, 提到120更稳)
    "steering_kd": 0.2,          # ★降低D项: 0.4放大振荡, 改0.2温和
    "decel_gain": 0.5,           # ★再降: 0.8仍全程D6, 改0.5让直道能上D8
    "speed_floor": 0.6,          # ★最低D6(0.6×10=6), 小车D6以下不动
    "speed_steer_gain": 0.10,    # ★降低: 0.20放大振荡, 改0.10
    "feedforward_gain": 0.0,     # ★禁用前馈: 0.15仍放大弯道突变, 改0
    "curvature_decel_gain": 0.5, # ★Phase3: 曲率减速降到0.5(2.0太激进, 和far_off叠加后全程D6)
    # ★赛道丢失恢复：蓝色赛道跑出视野时，不能直走出图！
    # 而是朝“最后一次看到赛道的方向”强力打方向，把赛道转回视野
    "arrow_lost_recover_gain": 0.2,  # 丢失时的固定转向强度(0.4太大满偏, 降到0.2温和找回赛道)
    "arrow_lost_speed": 0.6,         # ★丢失时D6(最低可动速度, 实际用_d6_ratio覆盖)
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
    "debug_mode": False,

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
    # 航点导航已去掉, 起点就是vcar_init_pose的位置
    "waypoint_start": None,  # None=不走航点直接对准
    "waypoint_arrive_dist": 0.15,         # 距目标小于此值=到达，切换赛道导航
    # 初始位姿 [x, y, z, pitch, yaw, roll]（绝对坐标）
    # ★重要: 用网页WASD把车开到赛道起点，记下网页显示的 X/Y/Z/YAW 填到这里
    # 否则车会从(0,0,0)原点开始，可能不在赛道上。None=从原点开始
    "vcar_init_pose": [0.85, 0.0, 0.85, 0.0, 0.0, 0.0],  # 起点位置[x,y,z,pitch,yaw,roll]

    # 是否启用OCR（需要安装PaddleOCR；未安装时自动降级为黄色路牌检测）
    "enable_ocr": False,  # ★纯循迹模式: 禁用OCR
    # 路牌转向（识别到指示牌→右转/左转，接入虚拟车 final_offset）
    "sign_turn_gain": 0.12,        # 路牌转向强度(0.18冲出赛道, 降到0.12温和右转)
    "sign_right_hold_time": 1.0,   # 右转持续1秒(0.3秒太短路口靠左立即干扰)
    "sign_straight_hold_time": 1.5, # 非右转直行1.5秒
    "sign_ocr_interval": 3.0,      # 大模型调用慢, 3秒一次防卡顿
    "ernie_access_token": "",  # 百度AI Studio access_token(erniebot SDK用)
    "qianfan_api_key": "",  # 千帆v2 API Key(调ernie-5.0)

    # ===== UART1 真实小车控制(ASCII指令协议) =====
    "uart1_enable": True,        # 是否启用UART1真实小车同步控制
    "uart1_port": "/dev/ttyUSB1",  # USB转串口设备路径(实际枚举为ttyUSB1, 非ttyUSB0)
    "uart1_baudrate": 115200,    # 波特率(文档要求115200)
    "uart1_max_duty": 10,        # ★纯循迹模式: D10拉满速度(第2-3圈)
    "uart1_servo_center": 90,    # 舵机中位角度(90=居中)
    "uart1_servo_range": 20,     # ★回退到20°(30°振荡幅度太大A60↔A120, 20°更稳)
    "uart1_servo_invert": True, # 舵机方向反转(False方向反了, 改回True)
    "uart1_servo_trim": 0,     # 舵机虚位补偿(-2导致offset=0时也右偏, 改0)

    # ===== 赛道几何前馈(状态机: 直道计距离, 弯道计角度) =====
    "track_geo_enable": True,     # 启用赛道几何前馈
    "track_geo_blend": 0.6,      # 直道: 60%几何+40%实时
    "track_geo_k_heading": 0.005, # 直道航向误差增益
    "track_straight_length": 2.2, # 直道长度(米)
    "track_curve_angle": 175.0,   # 弯道完成判定角度(175°=容差5°)
    "track_curve_offset": 0.12,   # 弯道恒定右转offset
    "track_curve_blend": 0.8,     # 弯道: 80%几何+20%实时
    "track_curve_decel": 0.1,     # 弯道减速: D6+0.1

    # ===== 路径记忆(第1圈D8学习, 第2-3圈D10回放, 总共3圈) =====
    "path_memory_enable": False,   # ★禁用路径记忆(改用赛道几何前馈, 更稳定)
    "path_memory_max_laps": 3,    # 总共跑3圈
    "path_memory_lap1_duty": 8,   # 第1圈速度D8
    "path_memory_lap2_duty": 10,  # 第2-3圈提速D10
    "path_memory_record_interval": 0.08,
    "path_memory_lap_threshold": 0.3,
    "path_memory_min_lap_distance": 8.0,
    "path_memory_blend": 0.5,
    "path_memory_lookahead": 8,

    # 是否显示可视化窗口
    "show_window": False,

    # ★定位模式: "aruco"=ArUco平替, "self"=虚拟车自算, "official"=官方定位
    "location_mode": "aruco",
    "location_udp_port": 9006,   # 接收定位数据的UDP端口
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
            min_pixels=config.get("arrow_min_pixels", 80),
            roi_margin_x=config.get("arrow_roi_margin_x", 0.0),
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
                # ★定位模式下禁止vcar发UDP(由定位模块替代发到9005)
                if config.get("location_mode", "self") in ("aruco", "official"):
                    self.vcar._send = lambda: None
                    print("[VCar] 定位模式: vcar._send()已禁用(由定位模块替代)")
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
                    servo_trim=config.get("uart1_servo_trim", 0),
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

        # ★定位模块: ArUco平替/官方定位
        self.location_mode = config.get("location_mode", "self")
        self._location_sock = None
        self._location_data = None  # 最新定位数据 {x, z, yaw}
        if self.location_mode in ("aruco", "official"):
            try:
                import socket as _sock
                self._location_sock = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
                self._location_sock.settimeout(0.001)  # 非阻塞
                self._location_sock.bind(("0.0.0.0", config.get("location_udp_port", 9006)))
                print(f"[Location] {self.location_mode}模式已启动, 监听UDP {config.get('location_udp_port', 9006)}")
            except Exception as e:
                print(f"[Location] UDP监听失败: {e}, 回退到self模式")
                self.location_mode = "self"
                self._location_sock = None

        # ★多线程控制: 控制线程50fps持续发指令
        self._latest_drive = None  # (mode, offset, speed)
        self._drive_lock = threading.Lock()

        # 初始化检测模型预热
        self._det_warmup = False
        self._seg_warmup = False

    def _control_thread(self):
        """★控制线程: 50fps, offset变化时发指令, 不变时100ms保活
        含预测插值: 视觉帧间用offset变化趋势线性外推, 提升有效转向更新频率"""
        _last_offset = 999
        _last_speed = 999
        _last_mode = ""
        _last_send_time = 0
        _prev_vis_offset = None    # 上一次视觉帧offset(预测用)
        _prev_vis_time = None
        _cur_vis_offset = None     # 当前视觉帧offset
        _cur_vis_time = None
        _last_reconnect_time = 0   # ★UART1上次重连尝试时间
        while self.running:
            if self._latest_drive is not None:
                with self._drive_lock:
                    mode, offset, speed = self._latest_drive
                # 检测视觉帧更新(offset变化=新视觉帧到达)
                if _cur_vis_offset is None or round(offset, 4) != round(_cur_vis_offset, 4):
                    _prev_vis_offset = _cur_vis_offset
                    _prev_vis_time = _cur_vis_time
                    _cur_vis_offset = offset
                    _cur_vis_time = time.time()
                if self.uart1:
                    _now = time.time()
                    # ★断连时每2秒尝试重连一次(不每帧重试防刷屏)
                    if not self.uart1.is_connected:
                        if _now - _last_reconnect_time > 2.0:
                            print("[UART1] 断连中, 尝试重连...")
                            self.uart1._reconnect()
                            _last_reconnect_time = _now
                        time.sleep(0.02)
                        continue
                    # 预测插值: 非STOP/STARTUP模式, 用offset变化率外推
                    _send_offset = offset
                    if (mode not in ("STOP", "STARTUP", "WAYPOINT") and
                            _prev_vis_offset is not None and
                            _prev_vis_time is not None and
                            _cur_vis_time is not None):
                        _dt_vis = max(_cur_vis_time - _prev_vis_time, 0.001)
                        _offset_rate = (_cur_vis_offset - _prev_vis_offset) / _dt_vis
                        _elapsed = _now - _cur_vis_time
                        if 0 < _elapsed < 0.15:  # 150ms内预测, 超时用原始值
                            _send_offset = offset + _offset_rate * _elapsed
                            _send_offset = max(-0.5, min(0.5, _send_offset))
                    _changed = (round(_send_offset, 3) != _last_offset or
                               round(speed, 3) != _last_speed or
                               mode != _last_mode)
                    _timeout = (_now - _last_send_time > 0.1)
                    if mode == "STOP":
                        if _last_mode != "STOP":
                            self.uart1.send_stop()
                            _last_mode = "STOP"
                            _last_send_time = _now
                    elif _changed or _timeout:
                        self.uart1.send_drive(_send_offset, speed)
                        _last_offset = round(_send_offset, 3)
                        _last_speed = round(speed, 3)
                        _last_mode = mode
                        _last_send_time = _now
            time.sleep(0.02)  # 50fps

    def _update_location(self):
        """从UDP接收定位数据, 更新vcar.pose(闭环核心)
        含航向滤波: 无论ArUco还是官方模块, 都做跳变拒绝+低通滤波"""
        if self._location_sock is None or self.vcar is None:
            return
        import json as _json
        try:
            while True:  # 读空缓冲区, 只留最新
                data, addr = self._location_sock.recvfrom(4096)
                msg = _json.loads(data.decode('utf-8'))
                if msg.get("type") == "robot_position":
                    pos = msg.get("pos", [0, 0, 0])
                    euler = msg.get("euler", [0, 0, 0])
                    raw_yaw = euler[1]
                    # ★航向滤波(通用: ArUco/官方模块都适用)
                    if not hasattr(self, '_filt_yaw'):
                        self._filt_yaw = raw_yaw
                    else:
                        _diff = raw_yaw - self._filt_yaw
                        while _diff > 180: _diff -= 360
                        while _diff < -180: _diff += 360
                        if abs(_diff) <= 45:  # 跳变拒绝: >45°视为误检
                            self._filt_yaw += 0.15 * _diff  # 低通: 15%新值
                        # else: 丢弃, 保持旧值
                    # 延迟补偿
                    now = time.time()
                    pred_x, pred_z = pos[0], pos[2]
                    if self._location_data is not None:
                        dt = now - self._location_data.get("timestamp", now)
                        if dt > 0.01:
                            vx = (pos[0] - self._location_data["x"]) / dt
                            vz = (pos[2] - self._location_data["z"]) / dt
                            delay = 0.065
                            pred_x = pos[0] + vx * delay
                            pred_z = pos[2] + vz * delay
                    self._location_data = {
                        "x": pred_x, "y": pos[1], "z": pred_z,
                        "pitch": euler[0], "yaw": self._filt_yaw, "roll": euler[2],
                        "timestamp": now
                    }
                    self.vcar.pose["x"] = pred_x
                    self.vcar.pose["y"] = pos[1]
                    self.vcar.pose["z"] = pred_z
                    self.vcar.pose["pitch"] = euler[0]
                    self.vcar.pose["yaw"] = self._filt_yaw  # ★用滤波后航向
                    self.vcar.pose["roll"] = euler[2]
        except Exception:
            pass

    def _track_geometry_control(self, x, z, yaw):
        """赛道几何辅助(感知触发): 用ArUco航向变化率检测弯道, 几何做辅助
        - 弯道检测: 航向连续变化>阈值 = 在弯道(不依赖HSV或坐标)
        - 直道: HSV感知主导, 航向修正
        - 弯道: 恒定右转 + HSV修正
        - 圈数: 航向变化累计180°×2 = 一圈
        返回: (offset, desired_heading, segment_name, is_curve, near_curve)
        """
        if not hasattr(self, '_tg_last_yaw'):
            self._tg_last_yaw = yaw
            self._tg_last_pos = (x, z)
            self._tg_curve_counter = 0
            self._tg_straight_counter = 0
            self._tg_in_curve = False
            self._tg_seg_dist = 0.0
            self._tg_lap_heading = 0.0
            self._tg_lap = 1
            print(f"[TrackGeo] 启动 heading={yaw:.0f}")

        dx = x - self._tg_last_pos[0]
        dz = z - self._tg_last_pos[1]
        dist_moved = float(np.sqrt(dx*dx + dz*dz))
        yaw_diff = yaw - self._tg_last_yaw
        while yaw_diff > 180: yaw_diff -= 360
        while yaw_diff < -180: yaw_diff += 360
        self._tg_last_pos = (x, z)
        self._tg_last_yaw = yaw
        self._tg_seg_dist += dist_moved
        self._tg_lap_heading += yaw_diff

        straight_len = self.config.get("track_straight_length", 2.2)
        curve_offset_val = self.config.get("track_curve_offset", 0.12)
        k_h = self.config.get("track_geo_k_heading", 0.005)

        # ===== 弯道检测: 航向连续变化(物理检测, 不依赖HSV) =====
        if abs(yaw_diff) > 1.5:
            self._tg_curve_counter += 1
            self._tg_straight_counter = 0
        else:
            self._tg_straight_counter += 1
            self._tg_curve_counter = max(0, self._tg_curve_counter - 1)

        # 状态切换: 连续5帧航向变化→弯道, 连续8帧稳定→直道
        if not self._tg_in_curve and self._tg_curve_counter >= 5:
            self._tg_in_curve = True
            self._tg_seg_dist = 0.0
            print(f"[TrackGeo] ->弯道 (航向变化触发)")
        elif self._tg_in_curve and self._tg_straight_counter >= 8:
            self._tg_in_curve = False
            self._tg_seg_dist = 0.0
            print(f"[TrackGeo] ->直道 (航向稳定触发)")

        # 圈数: 累计航向变化360° = 一圈
        if abs(self._tg_lap_heading) > 350:
            self._tg_lap += 1
            self._tg_lap_heading = 0.0
            print(f"[TrackGeo] 第{self._tg_lap}圈!")

        is_curve = self._tg_in_curve
        near_curve = (not is_curve and self._tg_seg_dist > straight_len - 0.5)

        if is_curve:
            segment = "弯道"
            desired_heading = yaw
            offset = curve_offset_val
        elif near_curve:
            # ★近弯道: 渐进加入右转(最后0.5m线性过渡)
            segment = "直道"
            desired_heading = self._tg_last_yaw
            _progress = (self._tg_seg_dist - (straight_len - 0.5)) / 0.5  # 0→1
            _progress = max(0.0, min(1.0, _progress))
            offset = curve_offset_val * _progress * 0.5  # 最多50%的弯道offset
        else:
            segment = "直道"
            desired_heading = self._tg_last_yaw
            offset = 0.0

        return offset, desired_heading, segment, is_curve, near_curve

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

    # ========== Task 1: 行人/car避让 ==========
    def task_avoid_person(self, det_frame, boxes, classes, scores):
        """
        行人/car避让决策（返回避让偏移量 + 速度比例）
        ★纯循迹模式: 禁用避障, 直接返回无避让
        """
        return 0.0, 1.0  # 无避让偏移, 全速

    def _task_avoid_person_orig(self, det_frame, boxes, classes, scores):
        """
        [原始避障逻辑-已禁用] 行人/car避让决策

        策略:
        - car: 检测到就向反方向转向避让, 避开后靠巡线自动回正
        - 行人: 太近→停车等; 中距离→减速通过; 远处→正常速度

        返回:
            avoid_offset: 避让转向偏移 (-0.5~+0.5)
            speed_ratio:  速度比例 (0=停, 0.5=减速, 1.0=正常)
        """
        h, w = det_frame.shape[:2]
        avoid_offset = 0.0
        speed_ratio = 1.0
        nearest_person = None  # (bottom_ratio, obj_offset, box_h_ratio)
        nearest_car = None

        if boxes is not None:
            for box, cl, score in zip(boxes, classes, scores):
                is_person = (cl == self.config["class_human"] and
                             score > self.config["person_conf_thresh"])
                is_car = (cl == self.config["class_car"] and score > 0.4)
                if not (is_person or is_car):
                    continue

                left, top, right, bottom = [int(b) for b in box]
                obj_cx = (left + right) / 2
                bottom_ratio = bottom / h
                box_h_ratio = (bottom - top) / h
                obj_offset = (obj_cx / w) - 0.5  # <0在左, >0在右

                # 过滤远处小目标(car用更早阈值, 移动目标需提前反应)
                _trig_y = self.config.get("car_trigger_y", 0.08) if is_car else self.config["avoid_trigger_y"]
                _min_h = self.config.get("car_min_box_h", 0.08) if is_car else self.config["avoid_min_box_h"]
                if box_h_ratio < _min_h:
                    continue
                if bottom_ratio < _trig_y:
                    continue

                # 画框
                color = (0, 0, 255) if is_person else (255, 128, 0)
                label = "PERSON" if is_person else "CAR"
                cv2.rectangle(det_frame, (left, top), (right, bottom), color, 3)
                cv2.putText(det_frame, f"{label} {score:.2f}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                info = (bottom_ratio, obj_offset, box_h_ratio)
                if is_person:
                    # 只有行人和赛道(蓝色)重叠时才记入避障: 检查行人框底部40%区域是否有蓝色像素
                    _ft = top + int((bottom - top) * 0.6)
                    _roi = det_frame[max(0, _ft):min(h, bottom), max(0, left):min(w, right)]
                    if _roi.size > 0:
                        _hsv = cv2.cvtColor(_roi, cv2.COLOR_BGR2HSV)
                        _bmask = cv2.inRange(_hsv, np.array(self.config["arrow_hsv_lower"]),
                                             np.array(self.config["arrow_hsv_upper"]))
                        if np.count_nonzero(_bmask) / _bmask.size > 0.05:
                            if nearest_person is None or bottom_ratio > nearest_person[0]:
                                nearest_person = info
                else:  # car
                    if nearest_car is None or bottom_ratio > nearest_car[0]:
                        nearest_car = info

        # ★car避让: 向反方向转向(car在右→左转, car在左→右转)
        if nearest_car is not None:
            bottom_ratio, obj_offset, box_h_ratio = nearest_car
            if (bottom_ratio >= self.config["avoid_stop_y"] or
                    box_h_ratio >= self.config["avoid_bigbox_h"]):
                # 太近→停车
                speed_ratio = 0.0
                avoid_offset = 0.0
            else:
                # 向反方向避让: -sign(obj_offset) = car在右→负→左转
                proximity = min(1.0, bottom_ratio / self.config["avoid_stop_y"])
                _car_gain = self.config.get("car_avoid_gain", 0.55)
                avoid_offset = -np.sign(obj_offset) * _car_gain * proximity
                if self.config.get("avoid_invert", False):
                    avoid_offset = -avoid_offset
                speed_ratio = min(speed_ratio, self.config.get("car_avoid_speed", 0.5))

        # ★行人避让: 连续3帧确认才停车(防误检持续停车)
        if not hasattr(self, '_person_confirm'):
            self._person_confirm = 0
        if nearest_person is not None:
            self._person_confirm += 1
        else:
            self._person_confirm = 0
        
        if nearest_person is not None and self._person_confirm >= 3:
            bottom_ratio, obj_offset, box_h_ratio = nearest_person
            if (bottom_ratio >= self.config.get("person_stop_y", 0.55) or
                    box_h_ratio >= self.config["avoid_bigbox_h"]):
                # 行人在赛道上且够近→停车等
                speed_ratio = 0.0
                avoid_offset = 0.0
            # else: 行人在赛道上但还远→正常行驶(不减速)
        # 未确认(<3帧)或行人不在赛道上→正常行驶(不减速)

        # 可视化
        if nearest_person is not None or nearest_car is not None:
            action = "STOP" if speed_ratio <= 0.0 else \
                     ("AVOID R" if avoid_offset > 0 else "AVOID L") if avoid_offset != 0 else "SLOW"
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
        ★纯循迹模式: 禁用OCR, 直接返回直行
        """
        return True, "straight"

    def _task_sign_recognition_orig(self, det_frame, boxes, classes, scores):
        """
        [原始OCR逻辑-已禁用] 路牌OCR识别
        每0.5秒用OpenCV检测路牌区域(快)，只在检测到路牌时才调用OCR(慢，阻塞几秒)
        """
        if not hasattr(self, "_sign_direction"):
            self._sign_direction = "straight"
            self._sign_hold_until = 0.0  # hold截止时间戳
            self._sign_last_ocr = 0.0  # 上次调用OCR的时间
            self._sign_last_detect = 0.0  # 上次检测路牌区域的时间
            self._sign_non_sign_count = 0      # 连续非路牌计数
            self._sign_detect_pause_until = 0.0  # 暂停检测截止时间

        h, w = det_frame.shape[:2]
        now = time.time()

        # 保持: 识别到方向后持续hold指定时间(right=0.5秒, 否则=2秒)
        if now >= self._sign_hold_until:
            self._sign_direction = "straight"

        # 连续3次非路牌后暂停检测30秒, 避免反复检测浪费资源
        if now < self._sign_detect_pause_until:
            cv2.putText(det_frame, f"SIGN:{self._sign_direction} [暂停检测]",
                        (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            return True, self._sign_direction

        # 每0.5秒检测路牌区域 + OCR调用后4秒冷却(路牌近了可重新OCR, 以最后一次为准)
        if (now >= self._sign_hold_until
                and now - self._sign_last_detect >= 0.5
                and now - self._sign_last_ocr >= 4.0):
            self._sign_last_detect = now

            # ★优先用目标检测结果找路牌(class_road_sign), 不限颜色
            sign_img = None
            sign_cls = self.config.get("class_road_sign", 13)
            if boxes is not None and len(boxes) > 0:
                for i, cls in enumerate(classes):
                    if int(cls) == sign_cls:
                        x1, y1, x2, y2 = boxes[i]
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w, int(x2)), min(h, int(y2))
                        if (x2 - x1) > 20 and (y2 - y1) > 20:  # 最小20x20像素
                            sign_img = det_frame[y1:y2, x1:x2].copy()
                            print(f"[Sign] 目标检测到路牌(class={sign_cls}) "
                                  f"[{x1},{y1},{x2},{y2}] score={scores[i]:.2f}")
                            break

            # 目标检测没找到 → 回退到颜色检测
            _sign_blue_ratio = 0
            if sign_img is None:
                top_third = det_frame[0:int(h * 0.55), :]
                _result = self._detect_sign_by_color(top_third)
                if _result is not None:
                    _sign_blue_ratio, sign_img = _result

            # 路牌较远时继续靠近再OCR(图片更清晰), 不停车不OCR
            _sign_too_far = False
            if sign_img is not None and _sign_blue_ratio > 0 and _sign_blue_ratio < 0.03:
                print(f"[Sign] 路牌较远(blue={_sign_blue_ratio:.1%}) → 继续靠近再OCR")
                _sign_too_far = True
                sign_img = None  # 不触发OCR, 继续行驶

            # 非路牌计数(连续3次后暂停检测, "路牌远"不算非路牌)
            if sign_img is None and not _sign_too_far:
                self._sign_non_sign_count += 1
                if self._sign_non_sign_count >= 3:
                    self._sign_detect_pause_until = now + 30
                    print(f"[Sign] 连续{self._sign_non_sign_count}次非路牌 → 暂停检测30秒")
            elif sign_img is not None:
                self._sign_non_sign_count = 0  # 检测到路牌, 重置计数

            if sign_img is not None:
                # 检测到路牌 → 停车调用OCR(阻塞几秒)
                self._sign_last_ocr = now  # 更新OCR时间, 10秒冷却
                print(f"[Sign] 检测到路牌 → 停车调用OCR")
                # ★设STOP模式: 控制线程保持停车, 防止OCR阻塞期间保活重发drive导致车跑走
                with self._drive_lock:
                    self._latest_drive = ("STOP", 0.0, 0.0)
                    self._stop_recovery = 5  # OCR后恢复时前5帧全速起步
                if self.uart1: self.uart1.send_stop()  # 立即发停车指令
                if self.ocr is not None:
                    new_dir, raw = self.ocr.ask_ernie_with_image(sign_img)
                    print(f"[Sign] OCR结果 → {new_dir} ({raw})")
                    cv2.putText(det_frame, f"SIGN:{new_dir}",
                                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    # 判断OCR是否可靠(VL看不清/降级OCR无意义文字)
                    _unreliable = any(kw in str(raw).lower() for kw in ["无法识别", "过暗", "模糊", "识别有误", "1234567890", "无（"])
                    if _unreliable:
                        # 看不清 → 不保持方向, 缩短冷却, 继续靠近再识别
                        print(f"[Sign] OCR看不清 → 继续靠近再识别")
                        self._sign_direction = "straight"
                        self._sign_hold_until = now
                        self._sign_last_ocr = now - 2.0  # 冷却缩短到2秒(原4秒), 路牌近了可重新OCR
                    elif new_dir == "right":
                        self._sign_direction = "right"
                        self._sign_hold_until = now + self.config.get("sign_right_hold_time", 0.5)
                        self._sign_last_ocr = now + 6.0  # 可靠→10秒冷却(4+6), 不再重复OCR
                    else:
                        self._sign_direction = "straight"
                        self._sign_hold_until = now + self.config.get("sign_straight_hold_time", 2.0)
                        self._sign_last_ocr = now + 6.0  # 可靠→10秒冷却(4+6), 不再重复OCR
                else:
                    self._sign_direction = "right"
                    self._sign_hold_until = now + self.config.get("sign_right_hold_time", 0.5)
                    print(f"[Sign] OCR不可用 → 默认right")

        _hold_left = max(0, self._sign_hold_until - now)
        cv2.putText(det_frame, f"SIGN:{self._sign_direction} h{_hold_left:.1f}s",
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
            if blue_ratio < 0.012:  # 最小1.2%(提高, 路牌更近才处理)
                continue
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            roi_gray = cv2.cvtColor(top_img[by:by+bh, bx:bx+bw], cv2.COLOR_BGR2GRAY)
            std = float(np.std(roi_gray))
            white_ratio = float(np.sum(roi_gray > 80)) / roi_gray.size
            if blue_ratio > best[0]:
                best = (blue_ratio, std, white_ratio)
            # 蓝块>1.5% + std>30 + 白字>3% + 宽>高 → 路牌(std>30, 真路牌32-71通过, 大门17-31部分通过但OCR看不清会跳过)
            if blue_ratio >= 0.015 and std > 30 and white_ratio > 0.03 and bw > bh:
                print(f"[Sign] 蓝块{blue_ratio:.1%} std{std:.0f} 白{white_ratio:.0%} → 路牌！")
                return blue_ratio, top_img[by:by+bh, bx:bx+bw].copy()
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

        # ★路径记忆状态
        self._path_memory = []           # [{x, z, offset, speed}, ...]
        self._lap_start_pos = None       # 起点坐标 {x, z}
        self._last_record_pos = None     # 上次记录的位置 (x, z)
        self._lap_distance = 0.0         # 本圈已行驶距离
        self._lap2_ready = False         # 路径记忆是否就绪
        self._path_mem_idx = 0           # 回放时的当前索引
        # ★赛道几何模式: 全程D10(不需要学习圈)
        if self.uart1 and self.config.get("track_geo_enable", False):
            self.uart1.max_duty = self.config.get("uart1_max_duty", 10)
            print(f"[TrackGeo] 赛道几何模式: D{self.uart1.max_duty} (无需学习圈)")
        elif self.uart1 and self.config.get("path_memory_enable", False):
            self.uart1.max_duty = self.config.get("path_memory_lap1_duty", 8)
            print(f"[PathMem] 第1圈学习模式: D{self.uart1.max_duty}")
        # ★动态speed_floor: 始终对应D6
        self._d6_ratio = 6.0 / (self.uart1.max_duty if self.uart1 else 10)

        # 启动对准阶段状态: 原地转到赛道在正前方再出发(防止开局横穿地图)
        self._startup_done = False
        self._waypoint_done = False
        self._startup_turned = 0.0

        # ★启动控制线程(50fps持续发指令, 帧间不浪费)
        self._ctrl_thread = threading.Thread(target=self._control_thread, daemon=True)
        self._ctrl_thread.start()

        while self.running:
            try:
                # 1. 读取AR视频帧
                frame = self.ar_reader.read_frame()
                if frame is None:
                    if self.config.get("show_window", False) and cv2.waitKey(1) == 27:
                        break
                    continue

                # 2. 预热模型（第一帧时）
                self._warmup(frame)

                # 3. 目标检测推理(暂时禁用, 只跑巡线)
                if not hasattr(self, '_frame_count'):
                    self._frame_count = 0
                    self._last_det_result = None
                    self._last_seg_result = None
                self._frame_count += 1

                # ★目标检测(每2帧一次, 避障需要)
                if self._frame_count % 2 == 1 or self._last_det_result is None:
                    det_result, det_flag = self.detector.infer(frame)
                    self._last_det_result = det_result
                else:
                    det_result = self._last_det_result
                if det_result is not None:
                    det_frame, boxes, classes, scores = det_result
                else:
                    det_frame = frame.copy()
                    boxes, classes, scores = None, None, None

                # 4. 语义分割推理(★仅可视化用, show_window=False时跳过省NPU提帧率)
                #    lane_offset未参与控制决策, 关闭不影响巡线/避障
                if self.config.get("show_window", False) and self.seg is not None:
                    _need_seg = (self._frame_count % 3 == 1 or self._last_seg_result is None)
                    if hasattr(self, '_prev_final') and abs(self._prev_final) > 0.2:
                        _need_seg = False  # 弯道跳过分割
                    if _need_seg:
                        seg_result, seg_flag = self.seg.infer(frame)
                        self._last_seg_result = seg_result
                    else:
                        seg_result = self._last_seg_result
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

                # 6. 蓝色箭头导航（本赛道主导航, 传入平滑后的速度做自适应前瞻）
                # ★低通滤波speed_ratio: 防止D7→D8一帧跳变导致死区/远带权重阶跃
                #   _smooth_sr变化率≤0.1/帧, 参数渐变不触发正反馈
                if not hasattr(self, '_smooth_sr'):
                    self._smooth_sr = self._d6_ratio  # 初始D6(开局渐进加速)
                arrow_offset, arrow_vis, arrow_valid, arrow_info = \
                    self.arrow_nav.navigate(frame, speed_ratio=self._smooth_sr)

                # 7. 边缘检测(★仅可视化用, show_window=False时跳过省CPU)
                #    结果未参与控制决策, 关闭不影响巡线/避障
                if self.config.get("show_window", False):
                    edge_vis, edge_mask, edge_lines = self.edge_detector.detect(frame)
                else:
                    edge_vis = frame.copy()  # 占位(不会被访问)

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
                    speed_ratio = self._d6_ratio  # 丢失时D6
                base_offset = max(-0.5, min(0.5, base_offset))

                # ★死区(速度自适应): 用平滑后speed_ratio, 防止阶跃跳变
                #   D6(sr=0.75)→死区0.08, D8(sr=1.0)→死区0.04
                _dz_base = self.config.get("arrow_dead_zone", 0.08)
                _dz = _dz_base * max(0.4, 1.5 - self._smooth_sr)
                if abs(base_offset) < _dz:
                    base_offset = 0.0

                # 8.5 base_offset低通滤波(抑制路口质心横摆)
                if not hasattr(self, '_smooth_base'):
                    self._smooth_base = 0.0
                _alpha = self.config.get("base_smooth_alpha", 1.0)  # 1.0=不滤波(PD的D项提供阻尼)
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
                    sign_offset = +self.config.get("sign_turn_gain", 0.25)
                # left/straight → sign_offset=0(直行, 正常巡线通过路牌区域)

                # 12.5 决策融合: 路牌转向优先（覆盖巡线/金币）；避让仍可叠加
                if abs(sign_offset) > 0.01:
                    final_offset = sign_offset + avoid_offset  # 路牌轻偏(不叠加base, 避免被巡线抵消)
                    speed_ratio = max(speed_ratio, 0.5)   # 路牌转向时保证车动着转
                elif abs(avoid_offset) > 0.05:
                    final_offset = avoid_offset + base_offset * 0.5  # 避让为主, 巡线占50%便于避开后回赛道(仅car生效, 行人avoid=0走else)
                else:
                    final_offset = base_offset + avoid_offset + coin_offset
                final_offset = max(-0.5, min(0.5, final_offset))

                # 12.7 横摆抑制 + 路口靠左
                if not hasattr(self, '_prev_final'):
                    self._prev_final = 0.0
                    self._prev_arrow = 0.0
                    self._intersection_hold = 0
                # 路口检测: arrow_offset符号反转 = 质心横摆 = 到路口了(蓝色像素>500才触发, 防赛道丢失时误判)
                if arrow_valid and (arrow_info or {}).get("pixels", 0) > 500 and self._prev_arrow * arrow_offset < -0.01:
                    self._intersection_hold = self.config.get("intersection_hold_frames", 30)
                    print(f"[VCar] 路口(arrow {self._prev_arrow:+.2f}→{arrow_offset:+.2f}) 靠左{self._intersection_hold}帧")
                self._prev_arrow = arrow_offset
                # 路口且没路牌且OCR hold期已过 → 默认靠左(OCR刚识别完2秒内不干扰, 让车按巡线走)
                if self._intersection_hold > 0 and abs(sign_offset) < 0.01 and (not hasattr(self, '_sign_hold_until') or time.time() >= self._sign_hold_until):
                    if abs(base_offset) < 0.15:  # offset小=路口模糊, 靠左
                        final_offset = self.config.get("intersection_left_bias", -0.2)
                    # offset大=明确弯道, 不靠左, 正常跟箭头
                    self._intersection_hold -= 1
                # 变化率限制(仅路牌转向不受限): 每帧offset变化≤max_delta
                if abs(sign_offset) < 0.01:
                    max_delta = self.config.get("offset_max_delta", 0.2)  # 0.2=平衡: 不振荡且能回正
                    if abs(final_offset - self._prev_final) > max_delta:
                        final_offset = self._prev_final + np.sign(final_offset - self._prev_final) * max_delta
                self._prev_final = final_offset
                final_offset = max(-0.5, min(0.5, final_offset))

                # ★开局渐进加速: 3秒内D6→D10, 直行不转(防止起步冲出)
                _run_elapsed = time.time() - self._run_start_time
                self._in_opening = False
                _ai = arrow_info or {}  # 始终初始化, 防开局跳过时报错
                if _run_elapsed < 3.0 or getattr(self, '_startup_just_done', False):
                    final_offset = 0.0
                    speed_ratio = self._d6_ratio + (1.0 - self._d6_ratio) * (_run_elapsed / 3.0)
                    self._startup_just_done = False
                    self._in_opening = True
                    with self._drive_lock:
                        self._latest_drive = ("STARTUP", 0.0, speed_ratio)

                # ===== Phase 3: 曲率减速(远带前瞻 + 曲率双因素) =====
                _far_off = abs(_ai.get("far_offset", 0.0)) if arrow_valid else 0.0
                _near_off = abs(_ai.get("near_offset", 0.0)) if arrow_valid else 0.0
                _curvature = abs(_ai.get("far_offset", 0) - _ai.get("near_offset", 0)) if arrow_valid else 0.0
                _k_decel = self.config.get("decel_gain", 1.5)
                _k_curv = self.config.get("curvature_decel_gain", 2.0)
                _speed_floor = self._d6_ratio  # 始终D6
                # 减速 = 前瞻减速 + 曲率减速, 取较大值
                speed_ratio = min(speed_ratio, max(_speed_floor,
                    1.0 - _k_decel * _far_off - _k_curv * _curvature))
                # ★低通滤波speed_ratio: 0.7旧+0.3新, 慢速恢复防弯道出口D7→D8跳变
                if not hasattr(self, '_smooth_sr'):
                    self._smooth_sr = self._d6_ratio
                self._smooth_sr = 0.7 * self._smooth_sr + 0.3 * speed_ratio

                # ===== Phase 2: 前馈控制(远带偏移→提前转向, 绕过死区) =====
                _feedforward = 0.0
                if arrow_valid and abs(_ai.get("far_offset", 0)) > 0.03:
                    _ff_gain = self.config.get("feedforward_gain", 0.35)
                    _feedforward = _ai["far_offset"] * _ff_gain * self._smooth_sr
                    _feedforward = max(-0.15, min(0.15, _feedforward))  # 限幅防过冲
                    final_offset = final_offset + _feedforward

                # ===== PD控制(修正版: D项在base_offset上, 限幅±0.08) =====
                if not hasattr(self, '_pd_prev_base'):
                    self._pd_prev_base = 0.0
                    self._pd_prev_time = time.time()
                _kd = self.config.get("steering_kd", 0.4)
                _now_pd = time.time()
                _dt = max(_now_pd - self._pd_prev_time, 0.01)
                _dt_norm = _dt / 0.05
                _deriv_base = (base_offset - self._pd_prev_base) / _dt_norm
                self._pd_prev_base = base_offset
                self._pd_prev_time = _now_pd
                if _run_elapsed >= 3.0:  # 开局3秒后才启用
                    _d_contrib = max(-0.08, min(0.08, _kd * _deriv_base))
                    final_offset = final_offset + _d_contrib
                    if abs(final_offset) < 0.15 and self._smooth_sr < 0.8:
                        final_offset = final_offset * 0.85
                    final_offset = max(-0.5, min(0.5, final_offset))

                # ★速度自适应增益(高速加重转向)
                _k_v = self.config.get("speed_steer_gain", 0.20)
                final_offset = final_offset * (1.0 + _k_v * self._smooth_sr)
                final_offset = max(-0.5, min(0.5, final_offset))

                # ★输出滤波: 已禁用(alpha=0) — 0.6会把车锁死在满偏位置无法回正
                if not hasattr(self, '_filt_offset'):
                    self._filt_offset = 0.0
                _filt_alpha = self.config.get("output_filter_alpha", 0.3)
                final_offset = _filt_alpha * self._filt_offset + (1 - _filt_alpha) * final_offset
                self._filt_offset = final_offset
                final_offset = max(-0.5, min(0.5, final_offset))

                # ★卡死检测: 连续N帧满偏 → 强制减速防冲出
                if not hasattr(self, '_full_steer_count'):
                    self._full_steer_count = 0
                if abs(final_offset) > 0.15:
                    self._full_steer_count += 1
                else:
                    self._full_steer_count = 0
                if self._full_steer_count > 15:
                    speed_ratio = min(speed_ratio, self._d6_ratio)  # D6
                    if self._full_steer_count == 16 or (self._full_steer_count > 16 and self._full_steer_count % 10 == 0):
                        print(f"[WARN] 满偏{self._full_steer_count}帧 → 强制减速D6防冲出")
                if self._full_steer_count > 35:
                    speed_ratio = 0.0
                    if self._full_steer_count == 36:
                        print(f"[WARN] 满偏停车(可能已冲出赛道)")
                        self._filt_offset = 0.0
                        self._prev_final = 0.0
                        self._full_steer_count = 0

                # 可视化（用最终偏移，画到箭头导航图上）
                steering = self.task_lane_following(arrow_vis, final_offset, plan_info)

                # ★12.55 更新实车定位(闭环核心: 用实车实际位置覆盖虚拟车pose)
                self._update_location()

                # ===== HSV循迹 + 弯道前馈(简单可靠, 以实际情况为准) =====
                if not self._in_opening:
                    _far = _ai.get("far_offset", 0) if arrow_valid else 0.0
                    _far_abs = abs(_far)
                    # ★弯道前馈: far_offset越大, 右转辅助越大(平滑, 无状态切换)
                    if _far_abs > 0.05:
                        _curve_bias = min(0.10, _far_abs * 0.25)  # far=0.4→bias=0.10
                        final_offset += _curve_bias
                    # ★弯道减速: far_offset大 = 前方有弯道 = 减速
                    if _far_abs > 0.10:
                        speed_ratio = min(speed_ratio, self._d6_ratio + 0.1)
                    final_offset = max(-0.5, min(0.5, final_offset))
                    if self._frame_count % 50 == 0:
                        print(f"[Track] far={_far:+.2f} off={final_offset:+.3f} spd={speed_ratio:.2f}")

                # ===== 路径记忆(已禁用, 改用赛道几何) =====
                if self.config.get("path_memory_enable", False) and self.vcar is not None:
                    _px = self.vcar.pose.get("x", 0)
                    _pz = self.vcar.pose.get("z", 0)

                    # 设置起点: 必须收到定位模块实际数据(非init_pose), 且启动对准完成
                    if self._lap_start_pos is None and self._startup_done and self._location_data is not None:
                        self._lap_start_pos = (_px, _pz)
                        print(f"[PathMem] 起点(定位模块): ({_px:.2f}, {_pz:.2f})")

                    if self._lap_start_pos is not None:
                        # 累计行驶距离
                        if self._last_record_pos is not None:
                            _dx = _px - self._last_record_pos[0]
                            _dz = _pz - self._last_record_pos[1]
                            self._lap_distance += float(np.sqrt(_dx*_dx + _dz*_dz))

                        if self.current_lap == 1:
                            # ===== 第1圈: 记录路径 =====
                            if self._last_record_pos is None:
                                self._last_record_pos = (_px, _pz)
                            else:
                                _ddx = _px - self._last_record_pos[0]
                                _ddz = _pz - self._last_record_pos[1]
                                _dist_moved = float(np.sqrt(_ddx*_ddx + _ddz*_ddz))
                                if _dist_moved >= self.config.get("path_memory_record_interval", 0.08):
                                    # ★不记录垃圾数据: |offset|>0.2说明车在振荡/冲出, 不录
                                    if abs(final_offset) <= 0.2:
                                        self._path_memory.append({
                                            "x": _px, "z": _pz,
                                            "offset": final_offset,
                                            "speed": speed_ratio
                                        })
                                    self._last_record_pos = (_px, _pz)

                            # 圈完成检测: 回到起点附近且跑了足够远
                            _sx = _px - self._lap_start_pos[0]
                            _sz = _pz - self._lap_start_pos[1]
                            _dist_to_start = float(np.sqrt(_sx*_sx + _sz*_sz))
                            if (_dist_to_start < self.config.get("path_memory_lap_threshold", 0.3) and
                                    self._lap_distance > self.config.get("path_memory_min_lap_distance", 3.0) and
                                    _run_elapsed > 10.0 and
                                    self._full_steer_count == 0 and  # ★车在正常状态才算圈完成
                                    len(self._path_memory) > 30):  # ★至少30个好点才算有效圈
                                self.current_lap = 2
                                self._lap2_ready = True
                                self._lap_distance = 0.0
                                self._path_mem_idx = 0
                                if self.uart1:
                                    self.uart1.max_duty = self.config.get("path_memory_lap2_duty", 10)
                                    self._d6_ratio = 6.0 / self.uart1.max_duty  # 更新D6比例
                                print(f"\n[PathMem] ★第1圈完成! 记录{len(self._path_memory)}个点")
                                print(f"[PathMem] 第2圈提速: D{self.uart1.max_duty if self.uart1 else '?'}")
                                print(f"[PathMem] 起点({_px:.2f},{_pz:.2f}) 圈完成\n")

                        elif self.current_lap >= 2 and len(self._path_memory) > 10:
                            # ===== 第2+圈: 回放路径 =====
                            # 找最近的记录点(简化: 从上次索引向前搜索, 不回头)
                            _best_idx = self._path_mem_idx
                            _best_dist = 1e9
                            _search_start = max(0, self._path_mem_idx - 2)
                            _search_end = min(len(self._path_memory), self._path_mem_idx + 20)
                            for _i in range(_search_start, _search_end):
                                _mp = self._path_memory[_i]
                                _ddx = _px - _mp["x"]
                                _ddz = _pz - _mp["z"]
                                _d = _ddx*_ddx + _ddz*_ddz
                                if _d < _best_dist:
                                    _best_dist = _d
                                    _best_idx = _i
                            self._path_mem_idx = _best_idx

                            # 前瞻N个点(提前转向)
                            _la_idx = min(_best_idx + self.config.get("path_memory_lookahead", 3),
                                          len(self._path_memory) - 1)
                            _mem_offset = self._path_memory[_la_idx]["offset"]
                            _mem_speed = self._path_memory[_la_idx]["speed"]

                            # ★不回放极端偏移: |mem_offset|>0.2说明记录时车在振荡, 用实时
                            if abs(_mem_offset) > 0.2:
                                _mem_offset = final_offset  # 回退到实时
                                _mem_speed = speed_ratio

                            # 混合: 70%记忆 + 30%实时
                            _blend = self.config.get("path_memory_blend", 0.7)
                            final_offset = _blend * _mem_offset + (1 - _blend) * final_offset
                            final_offset = max(-0.5, min(0.5, final_offset))
                            # 速度: 第2+圈不超记忆速度(D10比D8快, 弯道不能更快)
                            speed_ratio = min(speed_ratio, _mem_speed * 0.95)  # 比记忆慢5%更安全

                            # 圈完成检测
                            _sx = _px - self._lap_start_pos[0]
                            _sz = _pz - self._lap_start_pos[1]
                            _dist_to_start = float(np.sqrt(_sx*_sx + _sz*_sz))
                            _max_laps = self.config.get("path_memory_max_laps", 3)
                            if (_dist_to_start < self.config.get("path_memory_lap_threshold", 0.3) and
                                    self._lap_distance > self.config.get("path_memory_min_lap_distance", 5.0)):
                                self.current_lap += 1
                                self._lap_distance = 0.0
                                self._path_mem_idx = 0
                                if self.current_lap > _max_laps:
                                    # ★3圈跑完, 停车
                                    print(f"\n[PathMem] ★★ 全部{_max_laps}圈完成! 停车 ★★\n")
                                    speed_ratio = 0.0
                                    final_offset = 0.0
                                else:
                                    print(f"\n[PathMem] ★第{self.current_lap-1}圈完成! 开始第{self.current_lap}圈\n")

                            if self._frame_count % 50 == 0:
                                print(f"[PathMem] L{self.current_lap} idx={_best_idx}/{len(self._path_memory)} "
                                      f"mem_off={_mem_offset:+.3f} fin={final_offset:+.3f} "
                                      f"spd={speed_ratio:.2f} ({_px:.2f},{_pz:.2f})")

                # 12.6 驱动虚拟车体（final_offset控转向 + speed_ratio控速度）
                if self.vcar is not None and not self._in_opening:
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
                        aligned = arrow_valid  # ★检测到赛道就出发, 不原地对准(原地转画面不变=死循环)
                        timeout = abs(self._startup_turned) >= max_turn
                        if aligned or timeout:
                            self._startup_done = True
                            self._run_start_time = time.time()  # ★重置计时器: 开局保护从对准完成后开始
                            self._startup_just_done = True  # ★标记: 下一帧强制开局保护
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
                        _ai = arrow_info or {}
                        print(f"[VCar] {drive_mode:7s} "
                              f"base={base_offset:+.3f} avoid={avoid_offset:+.3f} "
                              f"coin={coin_offset:+.3f} sign={sign_offset:+.3f}({sign_dir[:1]}) "
                              f"final={final_offset:+.3f} "
                              f"spd={speed_ratio:.1f} | arrow_valid={arrow_valid} "
                              f"near={_ai.get('near_offset',0):+.3f}{'!' if not _ai.get('near_valid',True) else ''} "
                              f"far={_ai.get('far_offset',0):+.3f}{'!' if not _ai.get('far_valid',True) else ''} "
                              f"np={_ai.get('near_pixels',0)} fp={_ai.get('far_pixels',0)} "
                              f"| yaw={p['yaw']:6.1f}(d{dyaw:+.1f}) "
                              f"x={p['x']:+.2f} z={p['z']:+.2f} seq={self.vcar.seq}")

                    # ★虚实1:1同步(由控制线程50fps发送, offset变化时立即发)
                    with self._drive_lock:
                        if drive_mode == "STOP":
                            self._latest_drive = ("STOP", 0.0, 0.0)
                            self._stop_recovery = 5  # 恢复后前5帧全速起步
                        elif drive_mode == "STARTUP":
                            self._latest_drive = ("STARTUP", arrow_offset if arrow_valid else 0.3, 1.0)
                        elif drive_mode == "WAYPOINT":
                            self._latest_drive = ("WAYPOINT", offset, 1.0)
                        else:  # DRIVE / DRIVE*
                            # ★STARTUP刚完成/开局保护期: 用STARTUP模式绕过预测插值
                            if getattr(self, '_startup_just_done', False) or _run_elapsed < 3.0:
                                self._latest_drive = ("STARTUP", 0.0, speed_ratio)
                            else:
                                _recovery_spd = speed_ratio
                                if getattr(self, '_stop_recovery', 0) > 0:
                                    _recovery_spd = self._d6_ratio
                                    self._stop_recovery -= 1
                                self._latest_drive = ("DRIVE", final_offset, _recovery_spd)

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
                    if cv2.waitKey(1) == 27:  # ESC退出(仅显示时有效)
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

        # ★停车: S+D0 (v3速度环自动减速, 连发5次确保收到)
        if self.uart1 and self.uart1.is_connected:
            try:
                for _ in range(5):
                    self.uart1._send("S")
                    self.uart1._send("D0")
                self.uart1.serial_conn.flush()
                import time as _t; _t.sleep(0.1)
                print("[UART1] 已发送S+D0×5")
            except Exception as e:
                print(f"[UART1] 停车发送异常: {e}")

        self.serial.set_speed(0)
        self.serial.brake()

        # 释放资源
        if self.detector is not None:
            self.detector.release()
        if self.seg is not None:
            self.seg.release()
        if self.vcar is not None:
            self.vcar.close()
        self.ar_reader.close()
        self.serial.close()
        if self.uart1:
            self.uart1.close()

        cv2.destroyAllWindows()
        print("[INFO] 系统已停止")


# ============ 入口 ============
if __name__ == '__main__':
    controller = SmartCarController(config=CONFIG)
    controller.run()
