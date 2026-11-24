from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor
import sys

# cloudpickle 可能在某些环境缺失，降级为标准 pickle
try:
    import cloudpickle
except ImportError:
    import pickle as cloudpickle
    print("[WARN] cloudpickle 未安装，已降级为标准 pickle")


def initRKNN(rknnModel="yolov.rknn", id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        raise RuntimeError(f"加载RKNN模型失败 (ret={ret}): {rknnModel}\n"
                           f"可能原因: 1.模型文件损坏 2.文件传输时被文本模式破坏 3.模型与SDK版本不匹配")
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        raise RuntimeError(f"初始化NPU运行环境失败 (ret={ret})")
    print(rknnModel, "\t\tdone")
    return rknn_lite


def initRKNNs(rknnModel="yolo.rknn", TPEs=1):
    sys.modules['pickle'] = cloudpickle
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i % 3))
    return rknn_list


class rknnPoolExecutor():
    def __init__(self, rknnModel, TPEs, func):
        self.TPEs = TPEs
        self.queue = Queue()
        self.rknnPool = initRKNNs(rknnModel, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        # self.pool = ProcessPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, frame):
        self.queue.put(self.pool.submit(
            self.func, self.rknnPool[self.num % self.TPEs], frame))
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()