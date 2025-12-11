# 导入依赖库
import ttkbootstrap as ttk  # 增强版Tkinter，提供现代化UI组件
import sys  # 系统相关操作（如获取程序运行路径）
import os  # 文件路径与目录操作
import cv2  # OpenCV库，用于视频读取和图像处理
import numpy as np  # 数值计算库，用于数组操作
from ultralytics import YOLO  # YOLO目标检测模型库
import time  # 时间相关操作（如延时、计时）
import logging  # 日志记录，用于调试和错误追踪
from threading import Lock, Thread  # 线程锁和线程类，实现多线程处理
import threading  # threading线程
from queue import Queue  # 队列，用于线程间数据传递
import queue  # 队列相关异常处理
from tkinter import filedialog, messagebox  # Tkinter文件选择和消息弹窗
import configparser  # 配置文件解析器，读取set.ini配置
import concurrent.futures  # 高级线程池，管理多线程任务
import torch

# 读取配置文件的函数
def get_config_value(config, section, option, default):
    try:
        value = config.get(section, option)
        if isinstance(default, int):
            return int(value)
        elif isinstance(default, float):
            return float(value)
        return value
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        show_error(f"读取set.ini中 {section}.{option} 出错,将使用默认值 {default}")
        return default

# 射线法判断点是否在多边形内
def point_in_polygon(x, y, poly):
    """
    使用射线法判断点 (x,y) 是否在多边形 poly 内
    poly: [(x1,y1), (x2,y2), ...]
    返回: True/False
    """
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# 计算 IOU（交并比）
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area

# 检查目标是否已被捕获的函数
def is_target_not_yet_captured(captured_targets, target_identifier):
    return target_identifier not in captured_targets

# 保存检测截图的函数（全画面）
def save_detection_screenshot(original_frame, save_path, video_filename, frame_number, object_type, saved_objects_count, confidence):
    file_name = os.path.join(save_path,
                             f"{object_type}_{os.path.splitext(video_filename)[0]}_{saved_objects_count}.jpg")
    try:
        cv2.imwrite(file_name, original_frame)
        return True
    except Exception as e:
        logging.error(f"保存截图时出错: {e}")
        return False

# 显示错误信息的函数
def show_error(message):
    messagebox.showerror("错误", message)
    logging.error(message)

# 视频捕获上下文管理器类
class VideoCaptureContextManager:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()

# 视频处理核心类
class VideoProcessingCore:
    def __init__(self, progress_callback=None):
        self.paused = False
        self.stopped = False
        self.frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
        self.lock = Lock()
        self.pause_stop_lock = Lock()
        self.resume_event = threading.Event()
        self.resume_event.set()  # 初始为运行状态
        self.last_screenshot_time = {}
        self.last_screenshot = {}
        self.roi_points = []  #ROI 点（跨视频保留）
        self.progress_callback = progress_callback
        self.video_path_for_progress = None

        # 移动目标优化
        self.prev_positions = {}
        self.movement_buffer = {}
        self.stable_since = {}

    def detect_and_save(self, frames, original_frames, model, target_classes, class_mapping, current_times, save_path,
                        saved_objects, detection_info, only_movement_targets, captured_targets,
                        video_filename, fps, confidence_threshold):
        try:
            # 使用原始帧进行检测
            results = model(frames, verbose=False)

            for i, result in enumerate(results):
                boxes = result.boxes
                current_time = current_times[i]
                original_frame = original_frames[i]  #截图用全画面

                for box in boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    if cls_id in target_classes and confidence >= confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        object_key = f"{cls_id}_{x1}_{y1}_{x2}_{y2}"
                        object_type = class_mapping[cls_id]
                        current_box = (x1, y1, x2, y2)

                        # 移动判断（可选）
                        if only_movement_targets:
                            if not self.is_target_moving_enhanced(object_key, current_box, current_time):
                                continue

                        self.prev_positions[object_key] = current_box

                        # 使用纯 Python 射线法判断目标是否在 ROI 内
                        if self.roi_points:
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            try:
                                if not point_in_polygon(cx, cy, self.roi_points):
                                    continue  # 点不在 ROI 内
                            except Exception as e:
                                logging.warning(f"ROI 检测失败: {e}")
                                continue

                        target_identifier = f"{cls_id}_{round(x1 / 10)}_{round(y1 / 10)}_{round(x2 / 10)}_{round(y2 / 10)}"
                        if not is_target_not_yet_captured(captured_targets, target_identifier):
                            continue

                        if object_key not in saved_objects:
                            info = f"{object_type} 在 {current_time:.2f} 秒时出现在位置: ({x1}, {y1}), ({x2}, {y2})，置信度: {confidence:.2f}"
                            detection_info.append(info)
                            frame_number = int(current_time * fps)
                            if save_detection_screenshot(original_frame, save_path, video_filename, frame_number, object_type,
                                                         len(saved_objects), confidence):
                                saved_objects[object_key] = True
                                captured_targets.add(target_identifier)
                                self.last_screenshot_time[object_type] = current_time
                                #self.last_screenshot[object_type] = original_frame.copy()

            return results
        except Exception as e:
            show_error(f"模型推理出错: {e}")
            return None

    def is_target_moving_enhanced(self, object_key, current_box, current_time):
        prev_box = self.prev_positions.get(object_key)
        if prev_box is None:
            return True

        iou = calculate_iou(prev_box, current_box)
        if iou > movement_iou_threshold:
            if object_key not in self.stable_since:
                self.stable_since[object_key] = current_time
            return False

        x1, y1, x2, y2 = current_box
        x1_p, y1_p, x2_p, y2_p = prev_box
        w_curr = x2 - x1
        h_curr = y2 - y1
        w_prev = x2_p - x1_p
        h_prev = y2_p - y1_p
        w_avg = (w_curr + w_prev) / 2 + 1e-6
        h_avg = (h_curr + h_prev) / 2 + 1e-6

        dx = abs((x1 + x2) / 2 - (x1_p + x2_p) / 2)
        dy = abs((y1 + y2) / 2 - (y1_p + y2_p) / 2)

        rel_dx = dx / w_avg
        rel_dy = dy / h_avg

        moved = rel_dx > movement_relative_threshold or rel_dy > movement_relative_threshold

        if object_key not in self.movement_buffer:
            self.movement_buffer[object_key] = 0

        if moved:
            self.movement_buffer[object_key] += 1
        else:
            self.movement_buffer[object_key] = 0

        last_stable = self.stable_since.get(object_key, -10)
        recently_stable = (current_time - last_stable) < movement_stable_seconds
        if recently_stable and self.movement_buffer[object_key] >= movement_consecutive_frames:
            del self.stable_since[object_key]
            return True
        elif not recently_stable:
            return self.movement_buffer[object_key] >= movement_consecutive_frames

        return False

    def read_frames(self, video_path, speed_multiplier):
        logging.info(f"开始读取视频: {video_path}")
        with VideoCaptureContextManager(video_path) as cap:
            if not cap.isOpened():
                show_error("无法打开视频文件")
                return
            try:
                while True:
                    # 检查是否需要停止或暂停（无锁快检）
                    with self.pause_stop_lock:
                        if self.stopped:
                            break
                        should_wait = self.paused

                    if should_wait:
                        self.resume_event.wait()  # 阻塞直到 resume_event.set()
                        continue  # 回到循环开头重新检查 stopped

                    frames = []
                    original_frames = []
                    current_times = []

                    for _ in range(BATCH_SIZE):
                        for _ in range(int(speed_multiplier)):
                            if not cap.grab():
                                break
                        ret, frame = cap.retrieve()
                        if not ret:
                            break
                        current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        current_time = current_frame_idx / cap.get(cv2.CAP_PROP_FPS)
                        frames.append(frame)
                        original_frames.append(frame)
                        current_times.append(current_time)

                    if not frames:
                        break

                    put_success = False
                    for _ in range(3):
                        try:
                            self.frame_queue.put((frames, original_frames, current_times,
                                                os.path.basename(video_path), cap.get(cv2.CAP_PROP_FPS)), timeout=0.5)
                            put_success = True
                            break
                        except queue.Full:
                            time.sleep(0.01)
                            continue
                    if not put_success:
                        logging.debug(f"帧队列持续满，跳过 {len(frames)} 帧")

                    # 再次检查是否在读帧期间被停止
                    with self.pause_stop_lock:
                        if self.stopped:
                            break

            except Exception as e:
                show_error(f"读取帧时出错: {e}")
            finally:
                if not self.stopped:
                    self.frame_queue.put(None)
        logging.info(f"视频读取结束: {video_path}")

    def process_frames(self, model, target_classes, class_mapping, save_path, only_movement_targets, confidence_threshold):
        logging.info("开始处理帧")
        saved_objects = {}
        detection_info = []
        captured_targets = set()
        total_duration = None
        last_processed_time = 0.0

        try:
            while True:
                # 检查停止或暂停
                with self.pause_stop_lock:
                    if self.stopped:
                        break
                    should_wait = self.paused

                if should_wait:
                    self.resume_event.wait()
                    continue  # 重新检查 stopped

                try:
                    item = self.frame_queue.get(timeout=1)
                    if item is None:
                        break
                except queue.Empty:
                    continue  # 不 break，继续检查暂停/停止

                frames, original_frames, current_times, video_filename, fps = item

                self.detect_and_save(
                    frames, original_frames, model, target_classes, class_mapping,
                    current_times, save_path, saved_objects, detection_info,
                    only_movement_targets, captured_targets, video_filename, fps, confidence_threshold
                )

                if current_times:
                    last_processed_time = current_times[-1]

                if total_duration is None:
                    with VideoCaptureContextManager(self.video_path_for_progress) as cap:
                        if cap.isOpened():
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            total_duration = total_frames / fps if fps > 0 else 1.0
                        else:
                            total_duration = 100.0

                progress = min(100.0, (last_processed_time / total_duration) * 100)

                if self.progress_callback:
                    self.progress_callback(progress)

        except Exception as e:
            show_error(f"处理帧时出错: {e}")
        finally:
            if self.progress_callback and not self.stopped:
                self.progress_callback(100)
        logging.info("帧处理结束")

    def start_processing(self, video_path, model, target_classes, class_mapping, save_path, speed_multiplier,
                        only_movement_targets, start_button, roi_button, confidence_threshold):
        """
        启动单个视频的处理流程。此函数会阻塞，直到该视频处理完成或被用户停止。
        """
        self.video_path_for_progress = video_path

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_read = executor.submit(self.read_frames, video_path, speed_multiplier)
            future_process = executor.submit(self.process_frames, model, target_classes, class_mapping, save_path,
                                            only_movement_targets, confidence_threshold)

            try:
                concurrent.futures.wait([future_read, future_process], return_when=concurrent.futures.ALL_COMPLETED)
            except Exception as e:
                show_error(f"处理线程出错: {e}")



# YOLO类名列表
yolo_classes = [
    "人", "自行车", "小车", "摩托车", "飞机", "巴士", "火车", "货车", "船", "交通灯", "消防栓",
    "停车标志", "停车计费器", "长椅", "鸟", "猫", "狗", "马", "羊", "牛", "大象",
    "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带", "行李箱", "飞盘", "滑雪板",
    "滑雪橇", "运动球", "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子", "酒杯",
    "茶杯", "叉子", "刀", "勺子", "碗", "香蕉", "苹果", "三明治", "橙子", "西兰花",
    "胡萝卜", "热狗", "披萨", "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽植物", "床", "餐桌",
    "马桶", "电视", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱", "烤面包机",
    "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷"
]

# 检查多边形是否封闭的函数
def is_polygon_closed(roi_points):
    if len(roi_points) >= 3:
        return True
    return roi_points[0] == roi_points[-1]

# 检查多边形是否不交叉的函数
def is_polygon_non_crossing(roi_points):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    def do_intersect(p1, q1, p2, q2):
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True
        return False

    n = len(roi_points)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if do_intersect(roi_points[i], roi_points[(i + 1) % n], roi_points[j], roi_points[(j + 1) % n]):
                return False
    return True

# 视频处理GUI类
class VideoProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.core = VideoProcessingCore(progress_callback=self.update_progress)
        self.model = None
        self.video_paths = []
        self.current_video_index = 0
        self.processing_finished = False
        self.create_widgets()
        self.load_model()

    def update_progress(self, progress):
        self.progress_bar['value'] = progress
        self.root.update_idletasks()

    def create_widgets(self):
        self.video_frame = self.create_frame("选择视频文件:")
        self.video_frame.pack(pady=2)
        self.video_entry = self.create_entry(self.video_frame)
        self.create_button(self.video_frame, "浏览", self.select_video, "success")
        self.save_path_frame = self.create_frame("截图保存路径:")
        self.save_path_frame.pack(pady=2)
        self.save_path_entry = self.create_entry(self.save_path_frame)
        self.create_button(self.save_path_frame, "浏览", self.select_save_path, "info")
        self.object_frame = ttk.Frame(self.root)
        self.object_frame.pack(pady=2)
        self.checkbox_vars = {cls: ttk.BooleanVar() for cls in yolo_classes}
        for i in range(13):
            row_frame = ttk.Frame(self.object_frame)
            row_frame.pack(pady=2, anchor='w')
            for j in range(7):
                if i * 7 + j < len(yolo_classes):
                    cls = yolo_classes[i * 7 + j]
                    checkbox = ttk.Checkbutton(row_frame, text=cls, variable=self.checkbox_vars[cls])
                    checkbox.pack(side='left', padx=5)
        self.select_button = self.create_button(self.object_frame, "全选", self.toggle_select_all, "primary")
        self.speed_group_frame = ttk.Labelframe(self.root, text="分析倍速是使用丢帧方式实现，对快速移动目标视频慎用")
        self.speed_group_frame.pack(pady=5, fill='x', padx=5)
        self.speed_frame = ttk.Frame(self.speed_group_frame)
        self.speed_frame.pack(pady=2)
        self.speed_var = ttk.IntVar()
        self.speed_var.set(1)
        self.create_label(self.speed_frame, "选择分析倍速:")
        speeds = [1, 2, 4, 8, 16, 24, 32, 48, 64]
        for speed in speeds:
            radio = ttk.Radiobutton(self.speed_frame, text=f"{speed}X", variable=self.speed_var, value=speed)
            radio.pack(side='left', padx=2)
        self.roi_frame = ttk.Frame(self.root)
        self.roi_frame.pack(pady=2)
        self.roi_button = self.create_button(self.roi_frame, "选择关注区域", self.select_roi, "warning")
        self.roi_status_label = self.create_label(self.roi_frame, "没有选取关注区域", anchor='center')
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=2)
        self.start_button = self.create_button(self.button_frame, "开始处理", self.start_processing, "primary")
        self.pause_button = self.create_button(self.button_frame, "暂停处理", self.pause_processing, "secondary")
        self.create_button(self.button_frame, "结束处理", self.stop_processing, "danger")
        self.movement_frame = ttk.Frame(self.root)
        self.movement_frame.pack(pady=2)
        self.only_movement_var = ttk.BooleanVar(value=True)
        checkbox = ttk.Checkbutton(self.movement_frame, text="只对活动目标截图", variable=self.only_movement_var)
        checkbox.pack(side='left', padx=5)
        self.note_frame = ttk.Frame(self.root)
        self.note_frame.pack(pady=2)
        self.create_label(self.note_frame, "注意：分析过程中不要关闭程序，在弹出【视频分析已结束】窗口后再关闭。", anchor='center')
        Progress_frame = ttk.Frame(self.root)
        Progress_frame.pack(pady=0, padx=5, fill='x')
        self.progress_label = ttk.Label(Progress_frame, text="等待处理视频")
        self.progress_label.pack(pady=5, fill='x')
        self.progress_bar = ttk.Progressbar(Progress_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(pady=5, fill='x', expand=True)

    def create_frame(self, label_text):
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)
        self.create_label(frame, label_text)
        return frame

    def create_entry(self, frame):
        entry = ttk.Entry(frame, width=50)
        entry.pack(side='left', padx=5)
        return entry

    def create_button(self, frame, text, command, bootstyle):
        button = ttk.Button(frame, text=text, command=command, bootstyle=bootstyle)
        button.pack(side='left', padx=5)
        return button

    def create_label(self, frame, text, anchor=None):
        label = ttk.Label(frame, text=text)
        if anchor:
            label.pack(anchor=anchor)
        else:
            label.pack(side='left', padx=5)
        return label

    def select_video(self):
        try:
            self.video_paths = filedialog.askopenfilenames(filetypes=[("视频文件", "*.mp4;*.avi")])
            if self.video_paths:
                video_paths_str = ", ".join(self.video_paths)
                self.video_entry.delete(0, 'end')
                self.video_entry.insert(0, video_paths_str)
        except Exception as e:
            show_error(f"选择视频文件时出错: {e}")

    def select_save_path(self):
        try:
            save_path = filedialog.askdirectory()
            if save_path:
                self.save_path_entry.delete(0, 'end')
                self.save_path_entry.insert(0, save_path)
        except Exception as e:
            show_error(f"选择保存路径时出错: {e}")

    def get_selected_classes(self):
        return [text for text, var in self.checkbox_vars.items() if var.get()]

    def load_model(self):
        # 读取配置文件
        config = configparser.ConfigParser()
        config.read("set.ini")
        # 获取模型路径（从Model section读取，带默认值）
        model_path = get_config_value(config, "Model", "model_path", "./models/yolo11x.pt")

        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                show_error(f"模型文件不存在: {model_path}")
                return

            # 先加载模型到CPU
            model = YOLO(model_path)

            # 测试CUDA可用性并尝试迁移
            if torch.cuda.is_available():
                try:
                    # # 创建一个极小的 dummy 图像张量 (1x3x64x64)，并移动到 GPU
                    dummy_input = torch.zeros(1, 3, 64, 64).cuda()  # 小尺寸输入
                    model.model.cuda() # 确保模型在 GPU 上
                    _ = model.predict(dummy_input, device=0, verbose=False)
                    self.model = model
                    self.model.to('cuda:0')
                    logging.info(f"已启用CUDA加速，模型路径: {model_path}")
                    print(f"使用GPU推理，模型路径: {model_path}")
                except Exception as cuda_error:
                    logging.warning(f"CUDA推理失败，退回CPU: {cuda_error}")
                    self.model = YOLO(model_path)
                    self.model.to('cpu')
                    logging.info(f"使用CPU推理，模型路径: {model_path}")
            else:
                # 直接使用CPU
                self.model = model
                self.model.to('cpu')
                logging.info(f"CUDA不可用，使用CPU推理，模型路径: {model_path}")

        except Exception as e:
            show_error(f"模型加载失败: {e}")
            logging.error(f"模型加载失败: {e}")

    def start_processing(self):
        logging.info("开始处理视频")
        save_path = self.save_path_entry.get()
        selected_classes = self.get_selected_classes()
        speed_multiplier = self.speed_var.get()
        only_movement_targets = self.only_movement_var.get()

        if not self.video_paths or not save_path:
            show_error("请选择视频文件和截图保存路径")
            return
        if not selected_classes:
            show_error("请选择至少一个检测对象")
            return
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except Exception as e:
                show_error(f"创建保存路径时出错: {e}")
                return
        if self.model is None:
            show_error("模型加载失败，请检查模型文件是否存在")
            return

        class_mapping = {
            0: "人", 1: "自行车", 2: "小车", 3: "摩托车", 4: "飞机", 5: "巴士", 6: "火车", 7: "货车", 8: "船", 9: "交通灯", 10: "消防栓",
            11: "停车标志", 12: "停车计费器", 13: "长椅", 14: "鸟", 15: "猫", 16: "狗", 17: "马", 18: "羊", 19: "牛", 20: "大象",
            21: "熊", 22: "斑马", 23: "长颈鹿", 24: "背包", 25: "雨伞", 26: "手提包", 27: "领带", 28: "行李箱", 29: "飞盘", 30: "滑雪板",
            31: "滑雪橇", 32: "运动球", 33: "风筝", 34: "棒球棒", 35: "棒球手套", 36: "滑板", 37: "冲浪板", 38: "网球拍", 39: "瓶子", 40: "酒杯",
            41: "茶杯", 42: "叉子", 43: "刀", 44: "勺子", 45: "碗", 46: "香蕉", 47: "苹果", 48: "三明治", 49: "橙子", 50: "西兰花",
            51: "胡萝卜", 52: "热狗", 53: "披萨", 54: "甜甜圈", 55: "蛋糕", 56: "椅子", 57: "沙发", 58: "盆栽植物", 59: "床", 60: "餐桌",
            61: "马桶", 62: "电视", 63: "笔记本电脑", 64: "鼠标", 65: "遥控器", 66: "键盘", 67: "手机", 68: "微波炉", 69: "烤箱", 70: "烤面包机",
            71: "水槽", 72: "冰箱", 73: "书", 74: "时钟", 75: "花瓶", 76: "剪刀", 77: "泰迪熊", 78: "吹风机", 79: "牙刷"
        }
        target_classes = [key for key, value in class_mapping.items() if value in selected_classes]

        self.start_button.config(state='disabled')
        self.roi_button.config(state='disabled')
        self.pause_button.config(state='normal')
        self.current_video_index = 0
        self.processing_finished = False
        with self.core.pause_stop_lock:
            self.core.paused = False
            self.core.stopped = False

        def on_all_done():
            if not self.processing_finished:
                self.processing_finished = True

                def gui_update():
                    logging.info("视频分析已结束")
                    self.start_button.config(state='normal')
                    self.roi_button.config(state='normal')
                    self.progress_bar['value'] = 0
                    self.progress_label.config(text="等待处理视频")
                    messagebox.showinfo("提示", "视频分析已结束！")

                self.root.after(100, gui_update)

        def process_chain():
            total_videos = len(self.video_paths)
            while self.current_video_index < total_videos and not self.core.stopped:
                video_path = self.video_paths[self.current_video_index]
                video_filename = os.path.basename(video_path)

                self.progress_label.config(
                    text=f"正在处理 ({self.current_video_index + 1}/{total_videos}): {video_filename}"
                )
                self.progress_bar['value'] = 0
                logging.info(f"开始处理视频: {video_path}")

                try:
                    self.core.start_processing(
                        video_path=video_path,
                        model=self.model,
                        target_classes=target_classes,
                        class_mapping=class_mapping,
                        save_path=save_path,
                        speed_multiplier=speed_multiplier,
                        only_movement_targets=only_movement_targets,
                        start_button=self.start_button,
                        roi_button=self.roi_button,
                        confidence_threshold=confidence_threshold
                    )
                except Exception as e:
                    # 错误处理...直接跳过
                    pass
                finally:
                    self.current_video_index += 1

            self.root.after(100, on_all_done)

        Thread(target=process_chain, daemon=True).start()

    def pause_processing(self):
        with self.core.pause_stop_lock:
            self.core.paused = not self.core.paused
            if self.core.paused:
                self.core.resume_event.clear()  # 进入暂停
                self.pause_button.config(text="继续处理")
            else:
                self.core.resume_event.set()    # 恢复执行
                self.pause_button.config(text="暂停处理")

    def stop_processing(self):
        with self.core.pause_stop_lock:
            self.core.stopped = True
            self.core.paused = False
            self.core.resume_event.set()  # 唤醒等待中的线程
            self.pause_button.config(state='disabled')
            self.start_button.config(state='normal')
            self.roi_button.config(state='normal')
            self.current_video_index = 0

        # 清空队列
        while not self.core.frame_queue.empty():
            try:
                self.core.frame_queue.get_nowait()
            except:
                pass

    def toggle_select_all(self):
        all_selected = all(var.get() for var in self.checkbox_vars.values())
        if all_selected:
            for var in self.checkbox_vars.values():
                var.set(False)
            self.select_button.config(text="全选")
        else:
            for var in self.checkbox_vars.values():
                var.set(True)
            self.select_button.config(text="取消全选")

    def select_roi(self):
        try:
            if self.video_paths:
                cap = cv2.VideoCapture(self.video_paths[0])
                ret, frame = cap.read()
                cap.release()
                if ret:
                    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Select ROI", 800, 600)
                    roi_points = []
                    temp_frame = frame.copy()

                    def mouse_callback(event, x, y, flags, param):
                        nonlocal temp_frame
                        if event == cv2.EVENT_LBUTTONDOWN:
                            roi_points.append((x, y))
                            temp_frame = frame.copy()
                            for i, p in enumerate(roi_points):
                                cv2.circle(temp_frame, p, 5, (0, 0, 255), -1)
                                if i > 0:
                                    cv2.line(temp_frame, roi_points[i - 1], p, (0, 255, 0), 2)
                            if len(roi_points) > 1:
                                cv2.line(temp_frame, roi_points[-1], roi_points[0], (0, 255, 0), 2)
                        elif event == cv2.EVENT_RBUTTONDOWN and roi_points:
                            roi_points.pop()
                            temp_frame = frame.copy()
                            for i, p in enumerate(roi_points):
                                cv2.circle(temp_frame, p, 5, (0, 0, 255), -1)
                                if i > 0:
                                    cv2.line(temp_frame, roi_points[i - 1], p, (0, 255, 0), 2)
                            if len(roi_points) > 1:
                                cv2.line(temp_frame, roi_points[-1], roi_points[0], (0, 255, 0), 2)
                    cv2.setMouseCallback("Select ROI", mouse_callback)
                    while True:
                        cv2.imshow("Select ROI", temp_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('r'):
                            roi_points = []
                            temp_frame = frame.copy()
                        elif cv2.getWindowProperty("Select ROI", cv2.WND_PROP_VISIBLE) < 1:
                            if roi_points:
                                if is_polygon_closed(roi_points) and is_polygon_non_crossing(roi_points):
                                    self.core.roi_points = roi_points
                                    self.roi_status_label.config(text="已选取关注区域")
                                    self.roi_button.config(text="取消选取", command=self.cancel_roi)
                                else:
                                    show_error("选取的关注区域不合法，请确保区域封闭且不交叉。")
                            cv2.destroyAllWindows()
                            break
            else:
                show_error("请先选择视频文件")
        except Exception as e:
            show_error(f"选择关注区域时出错: {e}")

    def cancel_roi(self):
        self.core.roi_points = []
        self.roi_status_label.config(text="未选取关注区域")
        self.roi_button.config(text="选取关注区域", command=self.select_roi)


if __name__ == "__main__":
    # 读取配置文件
    config = configparser.ConfigParser()
    try:
        config.read('set.ini')
        BATCH_SIZE = get_config_value(config, 'Parameters', 'batch_size', 2)
        FRAME_QUEUE_SIZE = get_config_value(config, 'Parameters', 'frame_queue_size', 30)
        RESULT_QUEUE_SIZE = get_config_value(config, 'Parameters', 'result_queue_size', 15)
        target_moving = get_config_value(config, 'Parameters', 'target_moving', 5)
        confidence_threshold = get_config_value(config, 'Parameters', 'confidence_threshold', 0.3)

        movement_iou_threshold = get_config_value(config, 'target', 'movement_iou_threshold', 0.95)
        movement_relative_threshold = get_config_value(config, 'target', 'movement_relative_threshold', 0.02)
        movement_consecutive_frames = get_config_value(config, 'target', 'movement_consecutive_frames', 5)
        movement_stable_seconds = get_config_value(config, 'target', 'movement_stable_seconds', 5)

    except FileNotFoundError:
        show_error("未找到set.ini文件,将使用默认配置参数运行")
        BATCH_SIZE = 2
        FRAME_QUEUE_SIZE = 30
        RESULT_QUEUE_SIZE = 15
        target_moving = 5
        confidence_threshold = 0.3
        movement_iou_threshold = 0.95
        movement_relative_threshold = 0.02
        movement_consecutive_frames = 2
        movement_stable_seconds = 5

    # 配置日志
    logging.getLogger("ultralytics").setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='video_analysis.log', filemode='w')

    root = ttk.Window(themename='darkly')
    root.title("COCO数据集目标检测 by geckotao")

    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    try:
        from PIL import Image, ImageTk
        icon_path = os.path.join(base_path, 'icon.png')
        if os.path.exists(icon_path):
            icon_img = Image.open(icon_path)
            icon_photo = ImageTk.PhotoImage(icon_img)
            root.call('wm', 'iconphoto', root._w, icon_photo)
        else:
            print("图标文件icon.png不存在，使用默认图标")
    except Exception as e:
        print(f"设置窗口图标时出错: {e}")

    gui = VideoProcessorGUI(root)
    root.mainloop()
