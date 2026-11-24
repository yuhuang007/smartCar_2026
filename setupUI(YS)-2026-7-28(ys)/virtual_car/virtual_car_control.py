"""
虚拟车体控制模块
==================
通过 UDP 向 X-Verse AR 引擎（端口 9005）发送位姿指令，控制画面中的虚拟车体移动。
这是对官方网页 WASD 遥控功能的复刻：网页是"浏览器→Flask→UDP 9005"，
而本模块直接在板卡上"Python→UDP 9005"，用检测/巡线结果自动驾驶虚拟车。

【重要】端口 9005 的数据格式需要抓包确认:
    在板卡上执行: sudo tcpdump -i any -A -n 'udp port 9005'
    然后在网页按一下 W 键，观察打印出的真实数据包格式。
    如果格式和本模块的 _build_payload() 不一致，只需修改该方法即可。

坐标系说明（来自 main_config.json 的 axis_map）:
    axis_map_x = "-y", axis_map_z = "+x", axis_map_yaw = "-yaw"
    这些是引擎内部的轴映射，本模块只负责发送世界坐标位姿，映射由引擎处理。

运动模型（位姿式控制，类似WASD每次更新位置）:
    - 前进: x += speed * sin(yaw),  z += speed * cos(yaw)
    - 转向: yaw += steering_delta
    - 停车: speed = 0
"""

import socket
import json
import math
import time


class VirtualCarController:
    """虚拟车体控制器：通过UDP发送位姿，控制AR引擎中的虚拟车"""

    # 控制格式类型（抓包确认后选择正确的一种）
    FORMAT_JSON = "json"        # JSON格式（默认，最可能）
    FORMAT_CSV = "csv"          # 逗号分隔文本
    FORMAT_BINARY = "binary"    # 二进制浮点

    def __init__(self, target_ip="127.0.0.1", target_port=9005,
                 init_pose=None, fmt=FORMAT_JSON, ros_key=None,
                 sim_mode=False):
        """
        参数:
            target_ip:   AR引擎地址，板卡本地用 127.0.0.1
            target_port: 控制端口，默认9005（对应main_config.json的control_port）
            init_pose:   初始位姿 dict {x,y,z,pitch,yaw,roll}，None则用默认
            fmt:         数据格式，抓包确认后选择
            ros_key:     鉴权key（main_config.json里的ros_key，如引擎需要）
            sim_mode:    模拟模式，True则只打印不实际发送
        """
        self.target = (target_ip, target_port)
        self.fmt = fmt
        self.ros_key = ros_key
        self.sim_mode = sim_mode

        # 车体位姿状态（世界坐标，单位: 位置=米, 角度=度）
        # 初始y高度与拓包一致: pos[1]=0.0
        if init_pose is None:
            init_pose = {"x": 0.0, "y": 0.0, "z": 0.0,
                         "pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        self.pose = dict(init_pose)

        # 序列号（每发一包递增，与拓包看到的 seq 字段一致）
        self.seq = 0

        # 运动参数
        self.speed = 0.0            # 当前速度（米/步）
        self.max_speed = 0.05       # 最大速度（与网页速度滑块默认值一致）
        self.turn_rate = 5.0        # 转向速率（度/步，对应偏移量映射）

        # 创建UDP socket
        self.sock = None
        if not self.sim_mode:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                print(f"[VCar] UDP控制已就绪 → {self.target[0]}:{self.target[1]}")
            except Exception as e:
                print(f"[VCar] UDP创建失败，转模拟模式: {e}")
                self.sim_mode = True
        else:
            print("[VCar] 模拟模式（仅打印位姿，不实际发送）")

    def _build_payload(self):
        """
        构造发送给9005端口的数据包

        【已通过tcpdump抓包确认真实格式】
        真实格式示例:
        {"type": "robot_position", "is_sim": true,
         "pos": [x, y, z], "euler": [pitch, yaw, roll],
         "seq": 43, "timestamp": 1784272914.78}

        关键:
        - pos 是 [X, Y, Z] 数组
        - euler 是 [pitch, yaw, roll] 数组，yaw 在中间(index=1)
        - seq 递增，timestamp 用当前时间
        """
        self.seq += 1
        payload = {
            "type": "robot_position",
            "is_sim": True,
            "pos": [round(self.pose["x"], 6),
                    round(self.pose["y"], 6),
                    round(self.pose["z"], 6)],
            "euler": [round(self.pose["pitch"], 6),
                      round(self.pose["yaw"], 6),
                      round(self.pose["roll"], 6)],
            "seq": self.seq,
            "timestamp": time.time(),
        }
        return json.dumps(payload).encode("utf-8")

    def _send(self):
        """发送当前位姿到AR引擎"""
        payload = self._build_payload()
        if self.sim_mode or self.sock is None:
            print(f"[VCar-Sim] pose: x={self.pose['x']:.3f} z={self.pose['z']:.3f} "
                  f"yaw={self.pose['yaw']:.1f} speed={self.speed:.3f}")
            return
        try:
            self.sock.sendto(payload, self.target)
        except Exception as e:
            print(f"[VCar] 发送失败: {e}")

    # ========== 高层控制接口 ==========

    def drive(self, plan_offset, speed_ratio=1.0):
        """
        根据巡线偏移量驱动虚拟车前进（核心自动驾驶接口）

        参数:
            plan_offset: 路径规划偏移量 (-0.5~+0.5)，负=赛道偏左需左转
            speed_ratio: 速度比例 0~1（用于减速，如遇行人时传0.5）
        """
        # 1. 偏移量 → 转向: 偏移越大，转向角度越大
        steering_delta = plan_offset * self.turn_rate * 2
        self.pose["yaw"] += steering_delta
        self.pose["yaw"] %= 360.0

        # 2. 设置速度
        self.speed = self.max_speed * max(0.0, min(1.0, speed_ratio))

        # 3. 沿当前朝向前进（yaw转弧度）
        rad = math.radians(self.pose["yaw"])
        self.pose["x"] += self.speed * math.sin(rad)
        self.pose["z"] += self.speed * math.cos(rad)

        # 4. 发送位姿
        self._send()

    def stop(self):
        """停车：速度归零，位姿不变，持续发送保持当前位置"""
        self.speed = 0.0
        self._send()

    def turn_left(self, degrees=None):
        """原地左转"""
        self.pose["yaw"] -= (degrees if degrees else self.turn_rate)
        self.pose["yaw"] %= 360.0
        self._send()

    def turn_right(self, degrees=None):
        """原地右转"""
        self.pose["yaw"] += (degrees if degrees else self.turn_rate)
        self.pose["yaw"] %= 360.0
        self._send()

    def get_pose(self):
        """获取当前位姿"""
        return dict(self.pose)

    def close(self):
        """关闭socket"""
        if self.sock:
            self.sock.close()
            print("[VCar] UDP已关闭")


# ============ 独立测试 ============
if __name__ == "__main__":
    print("=" * 50)
    print("  虚拟车体控制测试")
    print("=" * 50)
    print("请先确认: 板卡上X-Verse系统已启动，且已激活遥控模式")
    print("真实格式已确认: {type, is_sim, pos[x,y,z], euler[pitch,yaw,roll], seq, timestamp}\n")

    # 从一个可见的起点开始（用网页当前看到的坐标，便于观察）
    # 如果车跳到奇怪位置，把下面改成网页实际显示的 X/Y/Z/YAW
    start_pose = {"x": 0.39, "y": 0.0, "z": 1.27,
                  "pitch": 0.0, "yaw": 22.0, "roll": 0.0}
    car = VirtualCarController(target_ip="127.0.0.1", target_port=9005,
                               init_pose=start_pose, sim_mode=False)

    print("\n--- 测试1: 直行前进5步 ---")
    for i in range(5):
        car.drive(plan_offset=0.0)   # 偏移0=直行
        time.sleep(0.2)

    print("\n--- 测试2: 向右巡线5步 ---")
    for i in range(5):
        car.drive(plan_offset=0.2)   # 正偏移=右转
        time.sleep(0.2)

    print("\n--- 测试3: 向左巡线5步 ---")
    for i in range(5):
        car.drive(plan_offset=-0.2)  # 负偏移=左转
        time.sleep(0.2)

    print("\n--- 测试4: 停车 ---")
    car.stop()

    print("\n最终位姿:", car.get_pose())
    car.close()
    print("测试完成!")
