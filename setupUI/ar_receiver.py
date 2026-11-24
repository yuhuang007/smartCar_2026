import time
import struct
import numpy as np
import cv2
from multiprocessing import shared_memory, resource_tracker
from infer_wrap import InferWrap

# ======= 新增：语义分割导航模块 =======
from road.seg_infer import SegModel
from road.seg_post import lane_offset_from_mask, overlay_mask

SHM_NAME = "shm_ar_video"
SHM_HEADER_SIZE = 16

#detecrion init
infer = InferWrap(TPEs=3)

# ======= 新增：语义分割初始化 =======
seg = SegModel()          # 加载 road/model/pp_liteseg.rknn
seg.init()                # 初始化 RKNN 运行时
RUN_MODE = "seg"          # "detect"=目标检测 | "seg"=语义分割

# ======= 新增：虚拟命令导航（后期替换为串口下发） =======
from road.nav_command import NavCommand
nav = NavCommand(dead_zone=20.0)

def remove_shm_from_resource_tracker():
    try:
        resource_tracker.unregister('/' + SHM_NAME, 'shared_memory')
    except:
        pass

def main():
    print("?? Client Ready. Waiting for Server...")
    
    while True:
        shm = None
        try:
            try:
                shm = shared_memory.SharedMemory(name=SHM_NAME)
                remove_shm_from_resource_tracker()
                print("? Connected!")
            except FileNotFoundError:
                time.sleep(1.0)
                continue

            last_fid = 0
            
            fps_t = time.time()
            fps_n = 0
            cur_fps = 0.0

            while True:
                try:
                    header = bytes(shm.buf[:SHM_HEADER_SIZE])
                    fid, w, h = struct.unpack('QII', header)
                    
                    if fid == last_fid:
                        time.sleep(0.002)
                        if cv2.waitKey(1) == 27: raise KeyboardInterrupt
                        continue

                    last_fid = fid
                    
                    size = w * h * 3
                    img_view = np.ndarray((h, w, 3), dtype=np.uint8,
                                          buffer=shm.buf[SHM_HEADER_SIZE:SHM_HEADER_SIZE+size])
                    frame = img_view.copy()
                    del img_view
                    
                    frame = cv2.flip(frame, 0)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    fps_n += 1
                    if time.time() - fps_t >= 1.0:
                        cur_fps = fps_n / (time.time() - fps_t)
                        fps_n = 0; fps_t = time.time()
                    
                    cv2.putText(frame, f"Client FPS: {cur_fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # ======= 新增：根据 RUN_MODE 切换检测/分割 =======
                    if RUN_MODE == "seg":
                        # 语义分割模式：输出 mask + 偏移量
                        mask = seg.predict_mask(frame, input_size=(512, 512), num_classes=2)
                        offset = lane_offset_from_mask(mask, roi_y_ratio=0.5)
                        vis = overlay_mask(frame, mask, alpha=0.45)
                        cv2.putText(vis, f"offset={offset:.1f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.imshow("ret", vis)
                        cmd = nav.decide(offset, fps=cur_fps)
                        # TODO: 后续把 offset 转成控制指令发送给下位机
                    else:
                        # 目标检测模式（原有逻辑）
                        det_frame, flag = infer.infer(frame)
                        cv2.imshow("ret", det_frame)
                    
                    if cv2.waitKey(1) == 27: raise KeyboardInterrupt

                except (ValueError, struct.error, BufferError):
                    raise FileNotFoundError

        except KeyboardInterrupt:
            print("\n?? Client Stop.")
            break
        except FileNotFoundError:
            print("?? Connection lost... Waiting...")
            if shm: shm.close()
            cv2.destroyAllWindows()
            time.sleep(1.0)
        finally:
            if shm: 
                try: shm.close()
                except: pass

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()