video_analysis

程序使用YOLO系列模型对监控视频进行目标推理分析，对符合的目标进行截图保存。

模型下载https://hf-mirror.com/Ultralytics/YOLO11/tree/main

请下载'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt', 'yolov12n.pt', 'yolov12s.pt', 'yolov12m.pt', 'yolov12l.pt', 'yolov12x.pt'放到models文件夹内

创建python虚拟环境

python -m venv mvenv

mvenv\Scripts\activate

安装依赖库

pip install opencv-python ultralytics ttkbootstrap lap psutil

运行

python main.py

[屏幕截图](https://github.com/user-attachments/assets/e4b1c230-86cb-4da1-9bbd-acb7f0b706c8)

