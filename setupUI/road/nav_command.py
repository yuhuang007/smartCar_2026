# road/nav_command.py
# 虚拟导航命令模块
# 功能：根据赛道中心偏移量，输出转向虚拟指令（LEFT/STRAIGHT/RIGHT）
# 说明：当前仅终端打印，后期替换为串口/USB-TTL下发给下位机

import time

# 偏移阈值（像素），可调参
DEAD_ZONE = 20.0

class NavCommand:
    def __init__(self, dead_zone=DEAD_ZONE):
        self.dead_zone = dead_zone
        self.last_cmd = "STRAIGHT"
        self.last_ts = time.time()

    def decide(self, offset: float, fps: float = 0.0):
        """
        输入：
            offset: 赛道中心偏移（>0 右偏，<0 左偏）
            fps: 当前帧率（可选，日志用）
        输出：
            cmd: "LEFT" / "STRAIGHT" / "RIGHT"
        """
        if offset < -self.dead_zone:
            cmd = "LEFT"
        elif offset > self.dead_zone:
            cmd = "RIGHT"
        else:
            cmd = "STRAIGHT"

        now = time.time()
        dt = now - self.last_ts
        self.last_cmd = cmd
        self.last_ts = now

        # 虚拟命令打印（后续替换为串口发送）
        line = f"[NAV] offset={offset:.1f} -> cmd={cmd} (fps={fps:.1f}, dt={dt*1000:.1f}ms)"
        print(line)

        return cmd