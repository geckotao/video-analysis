# video-analysis
COCO数据集目标检测
安装依赖库
pip install opencv-python ultralytics ttkbootstrap
要用CUDA加速请安装torch torchvision依赖库
使用免费开源的YOLO11模型对监控视频进行目标推理分析，对符合的目标进行截图保存到指定文件夹。
YOLO11模型会自动下载，也可自行下载（set.ini）设置模型名称及路径。
代码使用了AI进行优化。


set.ini 配置文件说明

set.ini 是程序的配置文件，用于调整目标检测和视频处理的相关参数。以下是各参数的详细说明：
[Parameters] 部分
batch_size：批处理大小，取值为正整数，默认值 4，每次模型推理处理的视频帧数，值越大处理速度越快，但占用内存也越多。
frame_queue_size：帧队列大小，取值为正整数，默认值 30，存储待处理视频帧的队列容量，需根据内存情况调整， 如日志文件出现“帧队列已满，跳过当前帧”就要调大（但要注意内存占用率，内存不足的要调少batch_size，避免内存堆积）。
result_queue_size：结果队列大小，取值为正整数，默认值 15，存储检测结果的队列容量，如日志文件出现“结果队列已满，跳过当前结”就要调大。
target_moving：目标移动检测参数，取值为正整数，默认值 5，预留参数，用于移动目标检测的基础阈值。
confidence_threshold：置信度阈值，取值范围 0.0-1.0，默认值 0.2，目标检测的置信度过滤阈值，值越高检测结果越严格（只有高置信度目标才会被保留）。

[target] 部分（移动目标检测相关）
movement_iou_threshold：移动 IOU 阈值，取值范围 0.0-1.0，默认值 0.95，用于判断目标是否静止的 IOU 阈值，超过此值认为目标未移动。
movement_relative_threshold：相对移动阈值，取值范围 0.0-1.0，默认值 0.02，目标中心相对移动距离的阈值，超过此值认为目标有移动。
movement_consecutive_frames：连续移动帧数，取值为正整数，默认值 5，判定目标为移动状态所需的连续帧数。
movement_stable_seconds：稳定时间阈值，取值为正整数，默认值 5，目标从静止状态变为移动状态所需的稳定时间（秒）。
