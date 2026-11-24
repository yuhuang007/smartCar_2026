"""
语义分割推理封装
类似于 infer_wrap/base/infer_wrap.py，但用于语义分割模型

功能：
1. 自动加载 road/model/ 下的 pp_liteseg.rknn 模型
2. 提供与 InferWrap 类似的接口
3. 支持多NPU核心并行推理
"""

import cv2
import os
import sys
import glob
import numpy as np

from rknnlite.api import RKNNLite

# 添加三个路径到Python搜索范围:
# 1. 项目根目录 (setupUI(YS)/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# 2. 当前目录 (seg_infer/base/) —— 找 seg_func.py
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# 3. infer_wrap/base/ 目录 —— 找 rknnpool.py (关键修复!)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "infer_wrap", "base")))

from rknnpool import rknnPoolExecutor
from seg_func import mySegFunc


def get_current_dir():
    return os.path.dirname(os.path.abspath(__file__))


class SegWrap():
    """
    语义分割推理封装类

    用法:
        from seg_infer import SegWrap
        seg = SegWrap(model_dir="road/model", TPEs=1)
        result_img, mask, lane_offset = seg.infer(frame)
    """

    def __init__(self, model_dir="road/model", TPEs=1):
        """
        参数:
            model_dir: 模型目录路径（相对于本文件或绝对路径）
            TPEs: 使用的NPU核心数 (1~3)
        """
        # 处理相对路径：相对于项目根目录
        if not os.path.isabs(model_dir):
            project_root = os.path.abspath(os.path.join(get_current_dir(), "..", ".."))
            model_dir = os.path.join(project_root, model_dir)

        model_path = self.get_model_path(model_dir)
        self.TPEs = TPEs
        self.rknn_pool = rknnPoolExecutor(
            rknnModel=model_path,
            TPEs=self.TPEs,
            func=mySegFunc
        )
        self.pool_flag = False

    def pool_init(self, img):
        """初始化线程池预热"""
        for i in range(self.TPEs + 1):
            self.rknn_pool.put(img)

    def get_model_path(self, model_dir):
        """自动查找目录下的 .rknn 文件"""
        models = glob.glob(os.path.join(model_dir, "*.rknn"))
        if len(models) == 0:
            raise FileNotFoundError(f"在 {model_dir} 中未找到 .rknn 模型文件")
        return models[0]

    def infer(self, img):
        """
        推理一帧图像

        参数:
            img: BGR格式图像 (numpy.ndarray)

        返回:
            result_img: 带分割可视化的图像
            mask: 分割标签图 (numpy.ndarray, 每个像素是类别ID)
            lane_offset: 赛道中心偏移量 (float, 正=偏右, 负=偏左)
        """
        if not self.pool_flag:
            self.pool_init(img)
            self.pool_flag = True
        self.rknn_pool.put(img)
        return self.rknn_pool.get()

    def __call__(self, *args, **kwds):
        return self.infer(*args, **kwds)

    def release(self):
        """释放NPU资源"""
        self.rknn_pool.release()


if __name__ == '__main__':
    # 测试代码
    seg = SegWrap("road/model", TPEs=1)

    # 用测试图片
    test_img = cv2.imread('bus.jpg')
    if test_img is not None:
        result, mask, offset = seg.infer(test_img)
        if result is not None:
            cv2.imwrite('seg_result.jpg', result)
            print(f"分割结果已保存，巡线偏移量: {offset:.3f}")
        seg.release()
    else:
        print("未找到测试图片")
