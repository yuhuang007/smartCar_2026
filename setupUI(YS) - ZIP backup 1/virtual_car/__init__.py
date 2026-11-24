"""
虚拟车体控制模块
通过UDP向X-Verse AR引擎(端口9005)发送位姿指令，控制虚拟车体移动
"""
from .virtual_car_control import VirtualCarController

__all__ = ["VirtualCarController"]
