# -*- coding: utf-8 -*-
"""
虚实小车运动校准脚本（测试完删除）
用法: python3 calibrate_test.py
功能: 分别让虚拟小车和真实小车执行相同指令, 测量运动差异, 计算校准参数
"""
import sys
import os
import time
import math

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from serial_comm.serial_control import UART1Controller
from virtual_car.virtual_car_control import VirtualCarController

# ============ 配置 ============
UART_PORT = "/dev/ttyUSB0"
UART_BAUD = 115200
MAX_DUTY = 17       # D15+B走 1.28m 不够, D17+B走 1.86m 太多, D16 估计~1.57m
SERVO_CENTER = 90
SERVO_RANGE = 25
SERVO_INVERT = True
SERVO_TRIM = -2     # 舵机虚位补偿(中间值: -1 不稳定，取 -2)
FPS = 30            # 假设帧率
TEST_SECONDS = 1.0  # 每次测试持续时间(秒)
TEST_FRAMES = int(TEST_SECONDS * FPS)  # 自动算帧数=30
DECEL_FRAMES = 3    # 最后 N 帧减速到 50%(减少惯性滑行)


def force_stop(uart1):
    """强制停车: B 倒转制动 + E 急停 + S 重置"""
    if not (uart1 and uart1.is_connected):
        print("[警告] UART1 未连接, 无法停车!")
        return
    # 第1阶段: B 切换倒车方向 + D20 提高倒车力度
    uart1._send("B")  # 先切换到倒车方向(用当前D11)
    uart1._send("D20")  # 再提高倒车占空比到20%(B已切换方向, 不会猛冲)
    try:
        uart1.serial_conn.flush()
    except:
        pass
    time.sleep(0.3)  # 300ms 倒转制动
    # 第2阶段: E 急停(切断电机)
    for _ in range(5):
        uart1._send("E")
    try:
        uart1.serial_conn.flush()
    except:
        pass
    time.sleep(0.02)
    # 第3阶段: S 重置状态 + D0 清零
    uart1._send("S")
    uart1._send("D0")
    try:
        uart1.serial_conn.flush()
    except:
        pass
    uart1._motor_running = False
    uart1._last_duty = 0


def run_test(uart1, vcar, offset, speed_ratio, label):
    """执行一次测试: 虚拟车+真实车同时运行TEST_FRAMES帧"""
    start_pose = dict(vcar.pose)
    print(f"\n{'='*50}")
    print(f"测试: {label}")
    print(f"参数: offset={offset}, speed_ratio={speed_ratio}, 时长={TEST_SECONDS}秒({TEST_FRAMES}帧)")
    print(f"虚拟车理论: 位移={0.05*speed_ratio*TEST_FRAMES:.3f}米, "
          f"角度变化={offset*10*TEST_FRAMES:.1f}度")
    print(f"{'='*50}")
    print("3秒后开始...")
    time.sleep(1)
    print("2秒...")
    time.sleep(1)
    print("1秒...")
    time.sleep(1)
    print("开始! (Ctrl+C可紧急停车)")

    try:
        for i in range(TEST_FRAMES):
            # 最后DECEL_FRAMES帧减速到50%(减少惯性滑行)
            if i >= TEST_FRAMES - DECEL_FRAMES:
                decel_progress = (i - (TEST_FRAMES - DECEL_FRAMES)) / DECEL_FRAMES
                actual_spd = speed_ratio * (1.0 - decel_progress * 0.5)  # 降到 50%
            else:
                actual_spd = speed_ratio
            vcar.drive(offset, actual_spd)
            if uart1 and uart1.is_connected:
                uart1.send_drive(offset, actual_spd)
            elapsed = (i+1) / FPS
            print(f"  运行中 {elapsed:.1f}s/{TEST_SECONDS}s", end="\r")
            time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        print("\n[!] 用户中断!")
    finally:
        # 立即强制停车
        print("\n[停车] 发送S+D0×15+flush...")
        force_stop(uart1)
        vcar.stop()
        print("[停车] 完成")

    end_pose = dict(vcar.pose)

    # 计算虚拟车实际位移
    dx = end_pose["x"] - start_pose["x"]
    dz = end_pose["z"] - start_pose["z"]
    dist = math.sqrt(dx * dx + dz * dz)
    dyaw = (end_pose["yaw"] - start_pose["yaw"] + 180) % 360 - 180

    print(f"\n--- 虚拟小车结果 ---")
    print(f"位移: {dist:.3f}米 (dx={dx:+.3f}, dz={dz:+.3f})")
    print(f"角度变化: {dyaw:.1f}度")
    print(f"终态: x={end_pose['x']:.3f} z={end_pose['z']:.3f} yaw={end_pose['yaw']:.1f}")

    return dist, dyaw


def main():
    print("=" * 50)
    print("  虚实小车运动校准工具")
    print("  测试完记得删除本脚本")
    print("=" * 50)

    # 初始化UART1
    print(f"\n[1] 连接UART1串口 {UART_PORT}...")
    uart1 = UART1Controller(
        port=UART_PORT, baudrate=UART_BAUD,
        max_duty=MAX_DUTY, servo_center=SERVO_CENTER,
        servo_range=SERVO_RANGE, servo_invert=SERVO_INVERT,
        servo_trim=SERVO_TRIM
    )
    if uart1.is_connected:
        print(f"[UART1] 已连接 {UART_PORT}")
    else:
        print(f"[UART1] 连接失败, 将只测虚拟车")

    # 初始化虚拟小车(模拟模式)
    print("[2] 初始化虚拟小车(模拟模式)...")
    vcar = VirtualCarController(sim_mode=True)
    print(f"[VCar] max_speed={vcar.max_speed}米/帧, turn_rate={vcar.turn_rate}")

    results = []

    while True:
        print(f"\n{'='*50}")
        print("菜单:")
        print("  1. 前进测试 (直行20帧, 虚拟车走1米)")
        print("  2. 右转测试 (offset=0.3, 虚拟车转60度)")
        print("  3. 左转测试 (offset=-0.3, 虚拟车转-60度)")
        print("  4. 大右转测试 (offset=0.5, 虚拟车转100度)")
        print("  5. 汇总校准结果")
        print("  0. 退出")
        print(f"已完成 {len(results)} 次测试")

        choice = input("\n选择: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            dist, dyaw = run_test(uart1, vcar, 0.0, 1.0, "直行前进")
            real_dist = input("请用尺子量真实小车走了多远(米, 如0.5): ").strip()
            try:
                real_dist = float(real_dist)
                results.append(("前进", 0.0, 1.0, dist, dyaw, real_dist, None))
                ratio = real_dist / dist if dist > 0.001 else 0
                print(f"速度比例: 真实/虚拟 = {real_dist:.3f}/{dist:.3f} = {ratio:.2f}")
                print(f"  → max_duty={MAX_DUTY} 时, 真实车走了虚拟车的 {ratio:.0%}")
            except ValueError:
                print("输入无效")
        elif choice == "2":
            dist, dyaw = run_test(uart1, vcar, 0.3, 1.0, "右转(offset=0.3)")
            real_angle = input("请量真实小车转了多少度(如45): ").strip()
            try:
                real_angle = float(real_angle)
                results.append(("右转", 0.3, 1.0, dist, dyaw, None, real_angle))
                ratio = real_angle / dyaw if dyaw != 0 else 0
                print(f"角度比例: 真实/虚拟 = {real_angle:.1f}/{dyaw:.1f} = {ratio:.2f}")
            except ValueError:
                print("输入无效")
        elif choice == "3":
            dist, dyaw = run_test(uart1, vcar, -0.3, 1.0, "左转(offset=-0.3)")
            real_angle = input("请量真实小车转了多少度(负数=左转, 如-40): ").strip()
            try:
                real_angle = float(real_angle)
                results.append(("左转", -0.3, 1.0, dist, dyaw, None, real_angle))
                ratio = real_angle / dyaw if dyaw != 0 else 0
                print(f"角度比例: 真实/虚拟 = {real_angle:.1f}/{dyaw:.1f} = {ratio:.2f}")
            except ValueError:
                print("输入无效")
        elif choice == "4":
            dist, dyaw = run_test(uart1, vcar, 0.5, 1.0, "大右转(offset=0.5)")
            real_angle = input("请量真实小车转了多少度(如80): ").strip()
            try:
                real_angle = float(real_angle)
                results.append(("大右转", 0.5, 1.0, dist, dyaw, None, real_angle))
                ratio = real_angle / dyaw if dyaw != 0 else 0
                print(f"角度比例: 真实/虚拟 = {real_angle:.1f}/{dyaw:.1f} = {ratio:.2f}")
            except ValueError:
                print("输入无效")
        elif choice == "5":
            print(f"\n{'='*50}")
            print("校准结果汇总")
            print(f"{'='*50}")
            print(f"当前参数: max_duty={MAX_DUTY}, servo_range={SERVO_RANGE}, "
                  f"wheelbase=0.15")
            print()

            for i, (label, offset, spd, v_dist, v_dyaw, r_dist, r_angle) in enumerate(results):
                print(f"测试{i+1}: {label} (offset={offset}, spd={spd})")
                print(f"  虚拟车: 位移={v_dist:.3f}米, 角度={v_dyaw:.1f}度")
                if r_dist is not None:
                    ratio = r_dist / v_dist if v_dist > 0.001 else 0
                    suggested_duty = int(MAX_DUTY / ratio) if ratio > 0.01 else MAX_DUTY
                    print(f"  真实车: 位移={r_dist:.3f}米")
                    print(f"  速度比例: {ratio:.2f} → 建议 max_duty={suggested_duty}")
                if r_angle is not None:
                    ratio = r_angle / v_dyaw if v_dyaw != 0 else 0
                    if ratio < 1.0:
                        suggested_wb = 0.15 / ratio
                        print(f"  真实车: 角度={r_angle:.1f}度")
                        print(f"  角度比例: {ratio:.2f} (转向不足) → 建议 wheelbase={suggested_wb:.3f}")
                    elif ratio > 1.0:
                        suggested_wb = 0.15 / ratio
                        print(f"  真实车: 角度={r_angle:.1f}度")
                        print(f"  角度比例: {ratio:.2f} (转向过度) → 建议 wheelbase={suggested_wb:.3f}")
                    else:
                        print(f"  真实车: 角度={r_angle:.1f}度")
                        print(f"  角度比例: 1.00 (完美匹配!)")
                print()

            print("=" * 50)
            print("参数调整指南:")
            print("  1. 真实车走得比虚拟车近 → 增大 max_duty")
            print("  2. 真实车走得比虚拟车远 → 减小 max_duty")
            print("  3. 真实车转向不足 → 增大 wheelbase (serial_control.py中)")
            print("  4. 真实车转向过度 → 减小 wheelbase")
            print("=" * 50)

    # 清理: 先强制停车再关串口
    force_stop(uart1)
    if uart1:
        uart1.close()
    print("\n校准结束, 记得删除本脚本")


if __name__ == "__main__":
    main()
