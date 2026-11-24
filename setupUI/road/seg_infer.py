# road/seg_infer.py
# 语义分割推理模块（RKNN）
# 说明：
# - 使用相对路径加载 road/model/pp_liteseg.rknn
# - 输入：BGR图像（OpenCV默认）
# - 输出：mask（0/1），以及概率图
# - 已加入 input/output details 打印，用于适配不同 rknn 导出格式

import os
import cv2
import numpy as np

# 默认模型相对路径（相对于项目根目录 setupUI/）
DEFAULT_MODEL_REL = "road/model/pp_liteseg.rknn"

def _abs_from_project_root(rel_path: str) -> str:
    """
    把相对于项目根目录的路径转为绝对路径。
    本文件位置：setupUI/road/seg_infer.py
    所以项目根目录 = 上一级目录
    """
    here = os.path.dirname(os.path.abspath(__file__))            # setupUI/road/
    project_root = os.path.abspath(os.path.join(here, ".."))    # setupUI/
    return os.path.join(project_root, rel_path)

# 尝试导入 rknnlite（仅在 orangePi 上可用）
try:
    from rknnlite.api import RKNNLite
except Exception:
    RKNNLite = None

class SegModel:
    def __init__(self, model_rel_path: str = DEFAULT_MODEL_REL):
        if RKNNLite is None:
            raise RuntimeError("rknnlite 不可用，请在 OrangePi (Ubuntu) 上运行")

        self.model_path = _abs_from_project_root(model_rel_path)
        self.rknn = None

    def init(self):
        """初始化 RKNN 模型（加载 + 运行时）"""
        rknn = RKNNLite()

        print(f"[SegModel] loading rknn: {self.model_path}")
        ret = rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"加载 RKNN 失败: {self.model_path}")

        # 初始化运行时，可根据 NPU 策略选择 CORE
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError("RKNN runtime 初始化失败")

        self.rknn = rknn
        print("[SegModel] init done")

        # 诊断：打印模型 input/output details
        try:
            input_details = self.rknn.get_input_details()
            print("[SegModel] input_details:", input_details)
        except Exception:
            pass
        try:
            output_details = self.rknn.get_output_details()
            print("[SegModel] output_details:", output_details)
        except Exception:
            pass

    def predict_prob(self, img_bgr, input_size=(384, 384), num_classes=2):
        """
        返回：
          prob: 可能是 [C,H,W] 或单通道 [H,W]
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_in = cv2.resize(img_rgb, input_size)

        # 这个模型日志已明确是 NCHW
        img_feed = np.transpose(img_in, (2, 0, 1))            # HWC -> CHW
        img_feed = np.expand_dims(img_feed, 0).astype(np.uint8)  # [1,C,H,W]

        print("[SegModel] feed shape:", img_feed.shape)

        outputs = self.rknn.inference(inputs=[img_feed])

        out = outputs[0]  # 当前模型输出: (1, 384, 384) 或 [1, C, H, W]
        print("[SegModel] raw out shape:", out.shape)

        # 适配常见输出维度
        if out.ndim == 4:
            # [1, C, H, W]
            prob = out[0]  # [C,H,W]
        elif out.ndim == 3:
            if out.shape[0] == 1:
                # [1, H, W] -> [H, W]
                prob = out[0]
            elif out.shape[2] == num_classes:
                prob = np.transpose(out, (2, 0, 1))  # [H,W,C]->[C,H,W]
            else:
                prob = out
        elif out.ndim == 2:
            prob = out
        else:
            prob = out

        return prob

    def predict_mask(self, img_bgr, input_size=(384, 384), num_classes=2, thresh=0.5, use_sigmoid=True):
        """
        返回与原图同尺寸的二值 mask（赛道区域=1，背景=0）
        """
        h, w = img_bgr.shape[:2]
        prob = self.predict_prob(img_bgr, input_size=input_size, num_classes=num_classes)

        # 取单通道分数图
        if prob.ndim == 3:
            prob2d = prob[0] if prob.shape[0] == 1 else (prob[1] if prob.shape[0] >= 2 else prob[0])
        elif prob.ndim == 2:
            prob2d = prob
        else:
            prob2d = prob

        # 调试打印：看原始值分布
        #print("[SegModel] prob2d min/max/mean:", float(np.min(prob2d)), float(np.max(prob2d)), float(np.mean(prob2d)))

        # 是否做 sigmoid（很多分割rknn输出是logits）
        if use_sigmoid:
            prob2d = 1.0 / (1.0 + np.exp(-prob2d))

        mask_small = (prob2d > thresh).astype(np.uint8)

        # resize 回原图尺寸
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask