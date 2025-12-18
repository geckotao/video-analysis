# video-analysis
COCO数据集目标检测
这是一款基于 YOLOv11 和 OpenCV 开发的视频目标检测与截图工具，具备可视化操作界面，支持批量视频处理、ROI 区域选择、移动目标过滤、多倍速分析等功能。工具能够自动检测视频中的指定目标，对符合条件的目标进行截图保存，并采用 ROI 首次进入去重策略减少重复截图，并支持 GPU 加速以提升处理效率。

安装依赖包
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126 
pip install ttkbootstrap opencv-python numpy ultralytics  configparser pillow 

 下载 YOLO 模型
创建models文件夹
下载 YOLOv11 模型文件（yolo11x.pt）到 models 文件夹
官方下载地址：https://github.com/ultralytics/assets/releases
或使用其他 YOLOv11 模型（yolo11n.pt/yolo11s.pt/yolo11m.pt/yolo11l.pt）

配置文件设置（可选）
程序会自动创建set.ini文件（如果不存在）

关键参数说明

[Parameters]
batch_size = 2             每次送入模型的帧数（增大可提升推理速度，但增加内存占用，而且算力会拖后腿；建议 1~8）
frame_queue_size = 15              视频读取队列最大容量（帧批次数量），防止内存溢出（8G内存不要超15）
target_moving = 5
confidence_threshold = 0.3   目标检测置信度阈值（0.0~1.0），低于此值的目标将被忽略
roi_iou_threshold = 0.20    ROI 内目标去重的 IoU 阈值。 值越小，减少移动中的同一目标重复截图，减少重复 ； 值越大，防误将不同目标合并，但移动中的同一目标重复截图多。推荐范围：0.20 ~ 0.40  （在单一目标高速移动场景下，降低 roi_iou_threshold 可提升去重效果。）

[target]
movement_iou_threshold = 0.90  两帧间目标框 IoU 超过此值，视为“静止”
movement_relative_threshold = 0.02  目标中心相对位移阈值（相对于目标宽高），超过则视为“移动”
movement_consecutive_frames = 5  	需连续多少帧检测到移动，才确认为目标“活动”
movement_stable_seconds = 5     目标静止超过此秒数后，再次移动需满足连续帧条件

[Model]
model_path = ./models/yolo11x.pt    YOLO 模型文件路径
