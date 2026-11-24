# 归档说明

这是一份从本地比赛工作目录整理的可克隆快照。保留主工程、早期实验、五份历史源码快照、模型、定位工具、数据和参考手册；原目录未移除或覆盖任何文件。

## 数据与模型

- `datalane/Images/` 和 `datalane/Annotations/` 中的 4,755 对文件按文件名配对，标注值见 [`label_mapping.txt`](../datalane/label_mapping.txt)。
- 原 `images/images/` 的 4,758 帧中，4,755 帧与配对原帧逐字节相同。其余 `frame_000974.jpg` 至 `frame_000976.jpg` 保存在 [`Unlabeled/`](../datalane/Unlabeled/)。两处合并即可得到原来的全部帧。
- 保留各版本的 `.rknn` 文件和原路径，避免历史脚本加载不到对应资源。根目录的 `model.rknn` 与 `setupUI/` 下的默认检测权重相同；不要仅凭 [`label_list.txt`](../label_list.txt) 推定所有权重都使用同一标签表。
- 原始 [`FPV_AR01.zip`](https://github.com/yuhuang007/smartCar_2026/releases/download/archive-2026/FPV_AR01.zip) 作为 Release 附件；解压后为单个 `FPV_AR01.mp4`。

## 未纳入 Git 的内容

| 内容 | 处理 |
| --- | --- |
| `images.zip`、`aruco.zip`、`setupUI(YS) (2).zip` | 保留已解包的数据或源码，省略重复压缩包和其中的生成产物 |
| `setupUI(YS)(1).zip` | 保存有差异的版本为 [`setupUI(YS) - ZIP backup 1/`](../setupUI%28YS%29%20-%20ZIP%20backup%201/)，不上传原压缩包 |
| `build/`、`dist/`、`__pycache__/`、工具缓存、生成日志和空目录 | 只留在原工作目录 |
| 设备绑定授权、硬编码 OCR 凭据 | 不上传授权文件；各版本源码中的凭据值已清空 |
| 个人证书照片 | 含个人身份信息，未上传 |
| `软件杯HUD备赛材料 527.zip` | 另行打包的 HUD 备赛资料和第三方演示组件，未纳入本竞赛车源码归档 |

应用模板保留在源码目录中，但原板卡发行包和第三方 AR 引擎并未随仓库发布，不能用此归档直接替代厂商安装包。
