"""
串口通信模块
用于X-01板卡与下位机(TC264/TC164)之间的通信

硬件连接:
    X-01板卡 USB口 → CH340 USB-TTL模块 → 下位机串口

通信协议（自定义帧格式，需与下位机代码配合）:
    帧头(0xAA) + 数据长度 + 指令类型 + 数据 + 校验和 + 帧尾(0x55)

指令类型:
    0x01: 速度控制 (data: 2字节速度值, 0~1000)
    0x02: 转向控制 (data: 2字节舵机角度, 500~2500)
    0x03: 制动指令 (data: 无)
    0x04: 状态查询 (data: 无, 下位机返回状态)
"""

import struct
import time
import threading

# 尝试导入 pyserial，如果没安装则自动进入模拟模式
try:
    import serial
    _HAS_PYSERIAL = True
except ImportError:
    _HAS_PYSERIAL = False
    print("[Serial] 警告: pyserial 未安装，将使用模拟模式")
    print("[Serial] 安装命令: pip3 install pyserial")


class SerialController:
    """
    串口控制器

    用法:
        ctrl = SerialController(port="/dev/ttyUSB0", baudrate=115200)
        ctrl.set_speed(500)      # 设置速度
        ctrl.set_steering(1500)  # 设置转向(居中)
        ctrl.brake()             # 制动
        ctrl.close()
    """

    # 指令类型常量
    CMD_SPEED = 0x01
    CMD_STEERING = 0x02
    CMD_BRAKE = 0x03
    CMD_QUERY = 0x04

    # 帧头帧尾
    FRAME_HEAD = 0xAA
    FRAME_TAIL = 0x55

    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, timeout=0.1):
        """
        参数:
            port: 串口设备路径
                   Linux板卡: /dev/ttyUSB0 或 /dev/ttyS5
                   Windows调试: COM3, COM4 等
            baudrate: 波特率（需与下位机一致）
            timeout: 读超时(秒)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self._lock = threading.Lock()  # 线程安全锁

        self._connect(timeout)

    def _connect(self, timeout=2.0):
        """连接串口"""
        # 如果pyserial没装，直接进入模拟模式
        if not _HAS_PYSERIAL:
            print("[Serial] pyserial未安装，自动进入模拟模式")
            self.is_connected = False
            return

        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout
            )
            self.is_connected = True
            print(f"[Serial] 已连接 {self.port} @ {self.baudrate}bps")
        except Exception as e:
            print(f"[Serial] 连接失败: {e}")
            print(f"[Serial] 将使用模拟模式（仅打印指令，不实际发送）")
            self.is_connected = False

    def _build_frame(self, cmd_type, data=b''):
        """
        构造通信帧

        帧格式: [0xAA] [长度] [指令类型] [数据...] [校验和] [0x55]
        """
        length = len(data) + 1  # 指令类型 + 数据
        frame = struct.pack('BB', self.FRAME_HEAD, length)
        frame += struct.pack('B', cmd_type)
        frame += data

        # 校验和：所有字节之和的低8位（不含帧头帧尾）
        checksum = sum(frame[1:]) & 0xFF
        frame += struct.pack('B', checksum)
        frame += struct.pack('B', self.FRAME_TAIL)

        return frame

    def _send_frame(self, cmd_type, data=b''):
        """发送一帧数据"""
        frame = self._build_frame(cmd_type, data)

        with self._lock:
            if self.is_connected and self.serial_conn:
                try:
                    self.serial_conn.write(frame)
                except Exception as e:
                    print(f"[Serial] 发送失败: {e}")
                    self.is_connected = False
            else:
                # 模拟模式：仅打印
                cmd_names = {
                    self.CMD_SPEED: "SPEED",
                    self.CMD_STEERING: "STEER",
                    self.CMD_BRAKE: "BRAKE",
                    self.CMD_QUERY: "QUERY"
                }
                name = cmd_names.get(cmd_type, f"CMD_{cmd_type:02X}")
                print(f"[Serial-Mock] {name}: data={data.hex() if data else 'none'}")

    def set_speed(self, speed):
        """
        设置速度

        参数:
            speed: int, 速度值 0~1000 (0=停止, 1000=最大速度)
        """
        speed = max(0, min(1000, int(speed)))
        data = struct.pack('<H', speed)  # 小端2字节
        self._send_frame(self.CMD_SPEED, data)

    def set_steering(self, angle):
        """
        设置转向角度

        参数:
            angle: int, 舵机角度 500~2500 (1500=居中, <1500左转, >1500右转)
        """
        angle = max(500, min(2500, int(angle)))
        data = struct.pack('<H', angle)
        self._send_frame(self.CMD_STEERING, data)

    def brake(self):
        """紧急制动"""
        self._send_frame(self.CMD_BRAKE)

    def query_status(self):
        """查询下位机状态"""
        self._send_frame(self.CMD_QUERY)
        if self.is_connected and self.serial_conn:
            # 尝试读取回复
            try:
                response = self.serial_conn.read(16)
                return response
            except:
                pass
        return None

    def close(self):
        """关闭串口连接"""
        if self.serial_conn:
            self.serial_conn.close()
            self.is_connected = False
            print("[Serial] 已断开")


# ============================================================
# UART1 控制器 — ASCII指令协议(驱动真实小车下位机)
# 指令: F=启动电机  S=停止  Dxx=占空比(0~100)  Axx=舵机角度(0~180)
# ============================================================
class UART1Controller:
    """
    UART1串口控制器(ASCII指令)
    与SerialController(二进制帧)并行使用, 专门对接UART1文档协议的下位机

    用法:
        uart1 = UART1Controller(port="/dev/ttyS1", baudrate=115200)
        uart1.send_drive(offset=0.0, speed_ratio=0.5)  # 直行50%速度
        uart1.send_drive(offset=+0.3, speed_ratio=0.8)  # 右转80%速度
        uart1.send_stop()                               # 停车
        uart1.close()
    """

    def __init__(self, port="/dev/ttyS1", baudrate=115200, timeout=0.05,
                 max_duty=25, servo_center=90, servo_range=25, servo_invert=False,
                 servo_trim=0):
        """
        参数:
            port: 串口设备路径 (UART1通常为 /dev/ttyS1)
            baudrate: 波特率 (文档要求115200)
            max_duty: 最大电机占空比%(建议10~20, 太快危险)
            servo_center: 舵机中位角度(90=居中)
            servo_range: 舵机偏移范围(±45度, 即45~135)
            servo_invert: 舵机方向反转(True=offset正值对应A值减小)
            servo_trim: 舵机虚位补偿(直线偏左用负值, 偏右用正值)
        """
        self.port = port
        self.baudrate = baudrate
        self.max_duty = max_duty
        self.servo_center = servo_center
        self.servo_range = servo_range
        self.servo_invert = servo_invert
        self.servo_trim = servo_trim
        self.serial_conn = None
        self.is_connected = False
        self._lock = threading.Lock()
        self._motor_running = False
        self._last_duty = -1
        self._last_angle = -1
        self._connect(timeout)

    def _connect(self, timeout=2.0):
        if not _HAS_PYSERIAL:
            print("[UART1] pyserial未安装, 跳过")
            return
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout
            )
            self.is_connected = True
            print(f"[UART1] 已连接 {self.port} @ {self.baudrate}bps")
        except Exception as e:
            print(f"[UART1] 连接失败: {e}")

    def _send(self, cmd):
        """发送ASCII指令(自动加\r\n)"""
        with self._lock:
            print(f"[UART1] {cmd}")  # 始终打印指令
            if self.is_connected and self.serial_conn:
                try:
                    self.serial_conn.write((cmd + "\r\n").encode())
                except Exception as e:
                    print(f"[UART1] 发送失败: {e}")
                    self.is_connected = False

    def send_drive(self, offset, speed_ratio):
        """
        同步驱动真实小车(与vcar.drive对应)
        offset: -0.5~+0.5 (转向偏移, +右-左)
        speed_ratio: 0.0~1.0 (速度比例)
        """
        # ★阶段4: 非线性映射(|offset|^1.5, 小偏差细调大偏差加重)
        # 系数4补偿幂函数衰减(|0.5|^1.5=0.354, ×4×16=22.7→clamp16满偏)
        if abs(offset) > 0.005:
            _sign = 1 if offset > 0 else -1
            steer_angle = round(_sign * abs(offset)**1.5 * self.servo_range * 4)
            if self.servo_invert:
                steer_angle = -steer_angle
            angle = self.servo_center + steer_angle + self.servo_trim
        else:
            angle = self.servo_center + self.servo_trim
        angle = max(self.servo_center - self.servo_range,
                    min(self.servo_center + self.servo_range, angle))

        # 速度目标等级: speed_ratio * max_duty (v3: D=速度目标, 速度环自动调整PWM)
        duty = int(speed_ratio * self.max_duty)
        duty = max(0, min(100, duty))

        # 每帧发A+D, F只在启动时发一次
        self._send(f"A{angle}")
        self._send(f"D{duty}")
        if duty > 0 and not self._motor_running:
            self._send("F")
            self._motor_running = True
        elif duty == 0 and self._motor_running:
            # 停车: S+D0 (速度环自动减速到0)
            self._send("S")
            self._send("D0")
            self._motor_running = False
            self._last_duty = 0
        self._last_angle = angle
        self._last_duty = duty

    def send_stop(self):
        """停车(S+D0+flush, 确保下位机收到)"""
        self._send("S")
        self._send("D0")
        if self.serial_conn:
            try:
                self.serial_conn.flush()
            except:
                pass
        self._motor_running = False
        self._last_duty = 0

    def send_pose_delta(self, dyaw, dist, speed_ratio=0.0):
        """
        根据虚拟小车的实际运动增量驱动真实小车(虚实同步)
        dyaw: yaw变化(度, +右转-左转)
        dist: 位移(米)
        speed_ratio: 虚拟小车速度比例(0~1)
        """
        # 舵机角度: 中位 + dyaw * gain
        yaw_gain = 5  # 1度yaw变化 → 5度舵机偏移(降低增益防冲出赛道)
        if self.servo_invert:
            angle = int(self.servo_center - dyaw * yaw_gain)
        else:
            angle = int(self.servo_center + dyaw * yaw_gain)
        angle = max(self.servo_center - self.servo_range,
                    min(self.servo_center + self.servo_range, angle))

        # 占空比: speed_ratio * max_duty
        duty = int(speed_ratio * self.max_duty)
        duty = max(0, min(100, duty))

        self._send(f"A{angle}")
        self._send(f"D{duty}")
        if duty > 0 and not self._motor_running:
            self._send("F")
            self._motor_running = True
        elif duty == 0 and self._motor_running:
            self._send("D0")
            self._send("F")  # 电磁制动
            self._last_duty = 0

    def send_turn(self, step, direction="right"):
        """
        原地转向(启动对准阶段, 与vcar.turn_right/left对应)
        step: 转向步长(度)
        direction: 'right' 或 'left'
        """
        if self.servo_invert:
            direction = "left" if direction == "right" else "right"
        if direction == "right":
            angle = int(self.servo_center + min(step * 2, self.servo_range))
        else:
            angle = int(self.servo_center - min(step * 2, self.servo_range))
        self._send(f"A{angle}")
        self._last_angle = angle
        # 低速转动
        if self._last_duty != 10:
            self._send("D10")
            self._last_duty = 10
        if not self._motor_running:
            self._send("F")
            self._motor_running = True

    def close(self):
        """关闭前连发S+D0确保停车"""
        if self.serial_conn:
            for _ in range(5):
                self._send("S")
                self._send("D0")
            try:
                self.serial_conn.flush()
                import time; time.sleep(0.2)
            except:
                pass
            self.serial_conn.close()
            self.is_connected = False
            print("[UART1] 已断开(S+D0×5)")


# ============ 模拟信号测试脚本 ============
def mock_test():
    """
    模拟信号测试：不连接下位机，仅打印控制指令
    用于调试阶段验证决策逻辑是否正确
    """
    print("=" * 50)
    print("  模拟信号测试模式 (无需下位机)")
    print("  将打印所有控制指令，验证决策逻辑")
    print("=" * 50)

    ctrl = SerialController(port="MOCK_PORT")  # 会自动进入模拟模式

    # 测试: 前进
    print("\n--- 测试1: 直行前进 ---")
    ctrl.set_speed(600)
    ctrl.set_steering(1500)  # 居中
    time.sleep(1)

    # 测试: 左转
    print("\n--- 测试2: 左转 ---")
    ctrl.set_steering(1200)  # 左转
    time.sleep(0.5)
    ctrl.set_steering(1500)  # 回正
    time.sleep(1)

    # 测试: 右转
    print("\n--- 测试3: 右转 ---")
    ctrl.set_steering(1800)  # 右转
    time.sleep(0.5)
    ctrl.set_steering(1500)  # 回正
    time.sleep(1)

    # 测试: 制动
    print("\n--- 测试4: 紧急制动 ---")
    ctrl.brake()
    time.sleep(1)

    # 测试: 停止
    print("\n--- 测试5: 停止 ---")
    ctrl.set_speed(0)

    ctrl.close()
    print("\n模拟测试完成!")


if __name__ == '__main__':
    mock_test()
