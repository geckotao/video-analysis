import tkinter as tk
from tkinter import filedialog, IntVar, messagebox
import cv2
import os
import threading
import queue
import time
import numpy as np
import configparser
import traceback
from PIL import Image, ImageTk
import sys
import datetime
import unicodedata
import re
# 导入ttkbootstrap核心组件
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.widgets.scrolled import ScrolledFrame, ScrolledText
from ttkbootstrap.dialogs import Messagebox

# 尝试导入 pynvml 用于 GPU 监控
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# 安全文件名清洗
def sanitize_filename(filename: str) -> str:
    cleaned = "".join(c for c in filename if unicodedata.category(c) not in ("Cc", "Cf") and c not in r'<>:"/\|?*')
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned[:100] or "unnamed"

# 全局异常捕获
def setup_exception_logging():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app_crash.log")
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write(f"崩溃时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
            f.write("="*50 + "\n")
        messagebox.showerror("程序崩溃", f"程序启动失败，详细日志已保存到:\n{log_path}\n请检查日志或联系开发者。")
        sys.exit(1)
    sys.excepthook = handle_exception
setup_exception_logging()


# 类别与颜色 
COCO_CLASS_MAPPING = {
    'person': '人', 'bicycle': '自行车', 'car': '小车', 'motorcycle': '摩托车',
    'airplane': '飞机', 'bus': '客车', 'train': '火车', 'truck': '货车', 'boat': '船',
    'traffic light': '交通灯', 'fire hydrant': '消防栓', 'stop sign': '停车标志',
    'parking meter': '停车计时器', 'bench': '长椅', 'bird': '鸟', 'cat': '猫', 'dog': '狗',
    'horse': '马', 'sheep': '羊', 'cow': '牛', 'elephant': '大象', 'bear': '熊',
    'zebra': '斑马', 'giraffe': '长颈鹿', 'backpack': '背包', 'umbrella': '雨伞',
    'handbag': '手提包', 'tie': '领带', 'suitcase': '行李箱', 'frisbee': '飞盘',
    'skis': '滑雪板', 'snowboard': '滑雪板', 'sports ball': '球类', 'kite': '风筝',
    'baseball bat': '棒球棒', 'baseball glove': '棒球手套', 'skateboard': '滑板',
    'surfboard': '冲浪板', 'tennis racket': '网球拍', 'bottle': '瓶子',
    'wine glass': '酒杯', 'cup': '杯子', 'fork': '叉子', 'knife': '刀', 'spoon': '勺子',
    'bowl': '碗', 'banana': '香蕉', 'apple': '苹果', 'sandwich': '三明治', 'orange': '橙子',
    'broccoli': '西兰花', 'carrot': '胡萝卜', 'hot dog': '热狗', 'pizza': '披萨',
    'donut': '甜甜圈', 'cake': '蛋糕', 'chair': '椅子', 'couch': '沙发', 'potted plant': '盆栽',
    'bed': '床', 'dining table': '餐桌', 'toilet': '马桶', 'tv': '电视', 'laptop': '笔记本电脑',
    'mouse': '鼠标', 'remote': '遥控器', 'keyboard': '键盘', 'cell phone': '手机',
    'microwave': '微波炉', 'oven': '烤箱', 'toaster': '烤面包机', 'sink': '水槽',
    'refrigerator': '冰箱', 'book': '书', 'clock': '时钟', 'vase': '花瓶', 'scissors': '剪刀',
    'teddy bear': '泰迪熊', 'hair drier': '吹风机', 'toothbrush': '牙刷'
}
COCO_CLASSES = list(COCO_CLASS_MAPPING.keys())
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 255),
    (255, 128, 0), (255, 0, 128), (0, 255, 128), (128, 255, 0), (128, 0, 255), (0, 128, 255)
]
NMS_IOU_THRESHOLD_DEFAULT = 0.45

# 导入Ultralytics、PyTorch依赖 
try:
    import torch
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    if torch.cuda.is_available():
        try:
            CUDA_STREAM = torch.cuda.Stream()
        except:
            CUDA_STREAM = None
    else:
        CUDA_STREAM = None
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    CUDA_STREAM = None
    Messagebox.showerror("依赖缺失", "缺少 Ultralytics 或 PyTorch！请安装：pip install ultralytics torch")

class VideoDetectionApp:
    def __init__(self, root: ttk.Window):
        self.root = root
        self.root.title("视频目标检测截图工具 by geckotao")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)

        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.ico")
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception as e:
                print(f"[WARN] 无法加载图标: {e}")
        self._log_buffer = []
        self._ui_ready = False
        self.log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "app_runtime.log")
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        self.save_folder = ""
        self.roi = None
        self.paused = False
        self.stopped = False
        self.model = None
        self.class_names = {i: name for i, name in enumerate(COCO_CLASSES)}
        self.class_vars = []
        self.current_processing_status = ""

        # 同步预览相关变量 
        self._preview_window = None
        self._preview_queue = queue.Queue(maxsize=10)
        self._preview_annotate_queue = queue.Queue(maxsize=2)
        self._preview_active = False
        self._preview_label = None  # 嵌入界面的预览标签（不再是窗口）
        self._current_preview_photo = None  # 保留预览图像引用，防止被回收
        self.current_frame = 0
        self.total_frames = 0
        self._current_file_idx = -1
        self.progress_lock = threading.Lock()
        self.paused_display = False 


        # 统计数据
        self.stats = {
            'fps': 0.0,
            'total_targets': 0,
            'targets_by_class': {},
            'gpu_mem': 'N/A'
        }

        # 多线程保存队列
        self.save_queue = queue.Queue(maxsize=100)
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        # 初始化 pynvml
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_handle = None
        else:
            self.gpu_handle = None

        if not self.load_configuration():
            return
        if not self.load_model():
            return
        self.setup_ui()
        self.update_stats_periodically()

    def _save_worker(self):
        """独立线程：异步保存截图"""
        while True:
            try:
                task = self.save_queue.get()
                if task is None:
                    break
                image, save_path = task
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(save_path)
                self.save_queue.task_done()
            except Exception:
                continue

    def load_configuration(self):
        config = configparser.ConfigParser()
        config_path = 'set.ini'
        if not os.path.exists(config_path):
            config['Settings'] = {
                'model_path': './models/yolo11x.pt',
                'nms_iou_threshold': str(NMS_IOU_THRESHOLD_DEFAULT),
                'inference_size': '640',
                'batch_size': '8',
                'roi_iou_threshold': '0.2',
                'movement_iou_threshold': '0.6',
                'movement_relative_threshold': '0.02',
                'movement_consecutive_frames': '5',
                'movement_stable_seconds': '5'
            }
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    config.write(f)
                self.log_message(f"已创建默认配置文件 {config_path}")
            except Exception as e:
                Messagebox.showerror("错误", f"无法创建配置文件: {e}")
                return False
        else:
            config.read(config_path, encoding='utf-8')

        self.MODEL_PATH = config.get('Settings', 'model_path', fallback='./models/yolo11x.pt')

        try:
            self.NMS_IOU_THRESHOLD = float(config.get('Settings', 'nms_iou_threshold', fallback=str(NMS_IOU_THRESHOLD_DEFAULT)))
            self.NMS_IOU_THRESHOLD = max(0.0, min(1.0, self.NMS_IOU_THRESHOLD))
        except ValueError:
            self.NMS_IOU_THRESHOLD = NMS_IOU_THRESHOLD_DEFAULT

        try:
            self.INFERENCE_SIZE = int(config.get('Settings', 'inference_size', fallback='640'))
            self.INFERENCE_SIZE = max(320, min(1280, self.INFERENCE_SIZE))
        except ValueError:
            self.INFERENCE_SIZE = 640

        try:
            self.BATCH_SIZE = int(config.get('Settings', 'batch_size', fallback='8'))
            self.BATCH_SIZE = max(1, self.BATCH_SIZE)
        except ValueError:
            self.BATCH_SIZE = 8

        try:
            self.ROI_IOU_THRESHOLD = float(config.get('Settings', 'roi_iou_threshold', fallback='0.2'))
        except ValueError:
            self.ROI_IOU_THRESHOLD = 0.2

        try:
            self.MOVEMENT_IOU_THRESHOLD = float(config.get('Settings', 'movement_iou_threshold', fallback='0.6'))
        except ValueError:
            self.MOVEMENT_IOU_THRESHOLD = 0.6

        try:
            self.MOVEMENT_RELATIVE_THRESHOLD = float(config.get('Settings', 'movement_relative_threshold', fallback='0.02'))
        except ValueError:
            self.MOVEMENT_RELATIVE_THRESHOLD = 0.02

        try:
            self.MOVEMENT_CONSECUTIVE_FRAMES = int(config.get('Settings', 'movement_consecutive_frames', fallback='5'))
        except ValueError:
            self.MOVEMENT_CONSECUTIVE_FRAMES = 5

        try:
            self.MOVEMENT_STABLE_SECONDS = float(config.get('Settings', 'movement_stable_seconds', fallback='5'))
        except ValueError:
            self.MOVEMENT_STABLE_SECONDS = 5

        self.log_message(
            f"配置加载成功: NMS IoU={self.NMS_IOU_THRESHOLD}, "
            f"推理分辨率={self.INFERENCE_SIZE}, batch_size={self.BATCH_SIZE}, "
            f"ROI IoU={self.ROI_IOU_THRESHOLD}"
        )
        return True

    def load_model(self):
        if not ULTRALYTICS_AVAILABLE:
            Messagebox.showerror("错误", "Ultralytics 未安装！")
            return False

        try:
            if not os.path.exists(self.MODEL_PATH):
                raise FileNotFoundError(f"模型文件不存在: {self.MODEL_PATH}")

            self.model = YOLO(self.MODEL_PATH)
            self.log_message("Ultralytics YOLO 模型加载成功！")
            return True
        except Exception as e:
            self.log_message(f"加载模型失败: {str(e)}")
            Messagebox.showerror("错误", f"模型加载失败:\n{str(e)}\n请确保模型由 Ultralytics 生成。")
            return False

    def _auto_adjust_batch_size(self):
        if not torch.cuda.is_available():
            self.log_message("未检测到 CUDA，使用 CPU 推理，batch_size 强制设为 1")
            self.BATCH_SIZE = 1
            return

        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception as e:
            self.log_message(f"获取 GPU 显存失败: {e}，默认设 batch_size=1")
            self.BATCH_SIZE = 1
            return

        model_name = os.path.basename(self.MODEL_PATH).lower()
        if 'yolo11n' in model_name:
            base_vram = 0.9
            model_type = "YOLO11n"
        elif 'yolo11s' in model_name:
            base_vram = 1.3
            model_type = "YOLO11s"
        elif 'yolo11m' in model_name:
            base_vram = 1.9
            model_type = "YOLO11m"
        elif 'yolo11l' in model_name:
            base_vram = 2.3
            model_type = "YOLO11l"
        elif 'yolo11x' in model_name:
            base_vram = 2.9
            model_type = "YOLO11x"
        else:
            base_vram = 2.9
            model_type = "Unknown"

        safe_vram_limit = total_vram_gb * 0.85
        if safe_vram_limit <= base_vram:
            max_batch = 1
        else:
            max_batch = 1 + int((safe_vram_limit - base_vram) / 0.03)
            max_batch = max(1, max_batch)

        original_batch = self.BATCH_SIZE
        self.BATCH_SIZE = min(original_batch, max_batch)

        self.log_message(
            f"GPU 显存: {total_vram_gb:.1f} GB | 模型: {model_type} | "
            f"inference_size: {self.INFERENCE_SIZE} → "
            f"自动设置 batch_size = {self.BATCH_SIZE} (用户配置: {original_batch})"
        )

    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=BOTH, expand=True)

        # 左侧选项设置框架
        left_frame = ttk.Labelframe(main_frame, text="选项设置", padding=10)
        left_frame.pack(side=LEFT, fill=Y, padx=(0, 10))

        # 视频文件选择
        self.video_entry = self._create_labeled_entry(left_frame, "选择视频文件:", "浏览", self.select_video)
        # 保存路径选择
        self.save_path_entry = self._create_labeled_entry(left_frame, "截图保存路径:", "浏览", self.select_save_path)

        # 目标类别选择框架（使用滚动框架）
        classes_frame = ttk.Labelframe(left_frame, text="选择目标类别", padding=10)
        classes_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        # 全选/取消按钮框架
        button_frame = ttk.Frame(classes_frame)
        button_frame.pack(pady=(0, 5), anchor=W)
        ttk.Button(button_frame, text="全选", command=self.select_all_classes).pack(side=LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="取消已选", command=self.deselect_all_classes).pack(side=LEFT)
        # 滚动画布
        scrollable_frame = ScrolledFrame(classes_frame, height=180)
        scrollable_frame.pack(side=LEFT, fill=BOTH, expand=True)
        # 类别复选框容器
        checkbox_container = ttk.Frame(scrollable_frame)
        checkbox_container.pack(fill=BOTH, expand=True)

        # 初始化类别复选框
        self.class_vars = []
        default_selected = {'bicycle', 'car', 'motorcycle', 'bus', 'truck'}
        for i, (idx, name) in enumerate(self.class_names.items()):
            row, col = divmod(i, 6)
            var = IntVar(value=1 if name in default_selected else 0)
            display = COCO_CLASS_MAPPING.get(name, name)
            cb = ttk.Checkbutton(checkbox_container, text=display, variable=var)
            cb.grid(row=row, column=col, sticky=W, padx=2, pady=1)
            self.class_vars.append(var)

        # 处理选项框架
        options_frame = ttk.Labelframe(left_frame, text="处理选项", padding=10)
        options_frame.pack(fill=X, pady=(0, 10))

        # 跳帧数设置
        ttk.Label(options_frame, text="跳帧数:", width=18).grid(row=0, column=0, sticky=W)
        self.speed_entry = ttk.Entry(options_frame)
        self.speed_entry.insert(0, "23")
        self.speed_entry.grid(row=0, column=1, sticky=EW, padx=5)
        ttk.Label(options_frame, text="(1=逐帧, 值越大越快)", foreground="gray").grid(row=0, column=2)

        # 置信度阈值设置
        ttk.Label(options_frame, text="置信度阈值:", width=18).grid(row=1, column=0, sticky=W)
        self.confidence_entry = ttk.Entry(options_frame)
        self.confidence_entry.insert(0, "0.1")
        self.confidence_entry.grid(row=1, column=1, sticky=EW, padx=5)
        ttk.Label(options_frame, text="(0.0-1.0)", foreground="gray").grid(row=1, column=2)

        # 只对移动目标截图
        self.only_moving_var = IntVar(value=1)
        ttk.Checkbutton(options_frame, text="只对移动目标截图", variable=self.only_moving_var).grid(row=2, column=0, sticky=W, padx=(0, 20))
        # 在截图中标注目标
        self.annotate_var = IntVar(value=0)
        ttk.Checkbutton(options_frame, text="在截图中标注目标", variable=self.annotate_var).grid(row=2, column=1, sticky=W)

        # ROI选择框架
        roi_frame = ttk.Frame(options_frame)
        roi_frame.grid(row=4, column=0, sticky=EW, pady=5, padx=(0, 10))  # 同一行row=4，第0列，右侧留间距
        self.roi_button = ttk.Button(roi_frame, text="选取关注区域", command=self.select_roi, bootstyle=INFO)
        self.roi_button.pack(side=LEFT)
        self.roi_label = ttk.Label(roi_frame, text="未选取关注区域", foreground="gray")
        self.roi_label.pack(side=LEFT, padx=5)

        # 预览框架（同一行号，分配第1列，取消跨列）
        preview_frame = ttk.Frame(options_frame)
        preview_frame.grid(row=4, column=1, sticky=EW, pady=5)  # 同一行row=4，第1列
        self.preview_button = ttk.Button(preview_frame, text="开启实时预览", command=self.toggle_preview, bootstyle=INFO)
        self.preview_button.pack(side=LEFT)
        self.preview_status = ttk.Label(preview_frame, text="预览: 关闭", foreground="gray")
        self.preview_status.pack(side=LEFT, padx=5)
        # 控制按钮框架
        control_button_frame = ttk.Frame(left_frame)
        control_button_frame.pack(pady=2)
        self.start_button = ttk.Button(control_button_frame, text="开始处理", command=self.start_processing)
        self.start_button.pack(side=LEFT, padx=5)
        self.pause_button = ttk.Button(control_button_frame, text="暂停处理", command=self.pause_processing, state=DISABLED, bootstyle=WARNING)
        self.pause_button.pack(side=LEFT, padx=5)
        self.stop_button = ttk.Button(control_button_frame, text="停止处理", command=self.stop_processing, state=DISABLED, bootstyle=DANGER)
        self.stop_button.pack(side=LEFT, padx=5)

        # 实时统计框架
        stats_frame = ttk.Labelframe(left_frame, text="实时统计", padding=10)
        stats_frame.pack(fill=X, pady=(10, 0))
        self.fps_label = ttk.Label(stats_frame, text="FPS(每秒处理帧数): 0.0")
        self.fps_label.pack(anchor=W)
        self.targets_label = ttk.Label(stats_frame, text="目标截图总数: 0")
        self.targets_label.pack(anchor=W)
        self.gpu_label = ttk.Label(stats_frame, text="占用 GPU 显存: N/A")
        self.gpu_label.pack(anchor=W)

        # 右侧处理状态框架
        right_frame = ttk.Labelframe(main_frame, text="处理状态", padding=10)
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # 状态标签
        self.status_label = ttk.Label(right_frame, text="等待开始......")
        self.status_label.pack(fill=X, pady=10)

        # 进度条
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(right_frame, variable=self.progress_var).pack(fill=X, pady=5)

        # 日志框架 
        log_frame = ttk.Labelframe(right_frame, text="处理日志", padding=10)
        log_frame.pack(fill=X, pady=(10, 5)) 
        # 1. 创建滚动条
        log_scrollbar = ttk.Scrollbar(log_frame, orient=VERTICAL)
        log_scrollbar.pack(side=RIGHT, fill=Y)

        # 2. 创建原生Text组件（支持state属性）
        self.log_text = tk.Text(log_frame,wrap=WORD,state=tk.DISABLED,height=8,yscrollcommand=log_scrollbar.set )
        self.log_text.pack(fill=BOTH, expand=True)

        # 3. 绑定滚动条与文本框
        log_scrollbar.config(command=self.log_text.yview)

        #预览框架
        preview_frame = ttk.Labelframe(right_frame, text="实时预览", padding=5)
        # 填充水平空间，垂直按需扩展，占据日志框下方区域
        preview_frame.pack(fill=BOTH, expand=True, pady=(5, 0))
        # 预览标签（用于显示视频帧），设置居中对齐
        self._preview_label = ttk.Label(preview_frame, anchor="center")
        self._preview_label.pack(fill=BOTH, expand=True)  # 填充预览框架空间


        self._ui_ready = True
        for msg in self._log_buffer:
            self.root.after_idle(lambda m=msg: self._safe_log(m))
        self._log_buffer.clear()

    def _create_labeled_entry(self, parent, label_text, button_text, command):
        frame = ttk.Frame(parent)
        ttk.Label(frame, text=label_text).pack(side=LEFT)
        entry = ttk.Entry(frame, width=50)
        entry.pack(side=LEFT, padx=5)
        ttk.Button(frame, text=button_text, command=command, bootstyle=PRIMARY).pack(side=LEFT)
        frame.pack(pady=2)
        return entry

    def log_message(self, message: str):
        # 1. 输出到控制台
        print(f"[LOG] {message}")
        
        # 2. 写入运行时日志文件
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        except Exception as e:
            print(f"[WARN] 写入日志文件失败: {e}")
        
        # 3. 输出到UI文本框
        if self._ui_ready:
            self.root.after(0, lambda msg=message: self._safe_log(msg))
        else:
            # 缓冲区仅作为兜底，同时打印缓冲区状态
            self._log_buffer.append(message)
            print(f"[BUFFER] UI未就绪，日志存入缓冲区，当前缓冲区长度: {len(self._log_buffer)}")

    def _safe_log(self, message: str):
        try:
            # 强制切换为可编辑状态
            if self.log_text["state"] != NORMAL:
                self.log_text.config(state=NORMAL)
            # 插入日志，添加时间戳
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(END, f"[{timestamp}] {message}\n")
            # 强制滚动到最新日志，避免日志被隐藏
            self.log_text.see(END)
            # 恢复为不可编辑状态
            self.log_text.config(state=DISABLED)
            # 刷新UI，确保日志立即显示
            self.log_text.update_idletasks()
        except Exception as e:
            # 若UI日志写入失败，输出到控制台和文件
            print(f"[WARN] UI日志写入失败: {e}，消息内容: {message}")
            # 写入日志文件备份
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [UI日志失败] {message}\n")
            except:
                pass

    def update_status(self, text: str):
        self.root.after_idle(lambda: self.status_label.config(text=text))

    def select_video(self):
        paths = filedialog.askopenfilenames()
        if paths:
            self.file_paths = list(paths)
            if len(paths) > 3:
                display_paths = ", ".join(paths[:2]) + f", ... (+{len(paths)-2} 个文件)"
            else:
                display_paths = ", ".join(paths)
            self.video_entry.delete(0, END)
            self.video_entry.insert(0, display_paths)
            self.log_message(f"已选择 {len(self.file_paths)} 个视频文件")

    def select_save_path(self):
        folder = filedialog.askdirectory()
        if folder:
            self.save_folder = folder
            self.save_path_entry.delete(0, END)
            self.save_path_entry.insert(0, self.save_folder)
            try:
                test = os.path.join(folder, "测试写入权限.tmp")
                open(test, 'w').close()
                os.remove(test)
                self.log_message("路径验证: 具有写入权限，支持中文路径")
            except Exception as e:
                Messagebox.showwarning("权限警告", f"保存路径可能有问题: {e}")

    #ROI---------------------------           
    #ROI 选择
    def select_roi(self):
        if not self.file_paths:
            self.log_message("错误: 请先选择视频文件")
            return

        cap = cv2.VideoCapture(self.file_paths[0])
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.log_message("错误: 无法读取视频首帧")
            return

        self._roi_frame_orig = frame.copy()
        self._roi_orig_h, self._roi_orig_w = frame.shape[:2]

        # 初始化缩放后的图像（先按原始比例缩放到合适大小）
        self._resize_roi_display(800, 600)  # 初始窗口大小

        # 创建窗口
        self._roi_window = ttk.Toplevel(title="选择关注区域 (ROI)       点完成【确认】设置关注区域，点【取消】或直接关闭窗口取消设置。")
        self._roi_window.geometry("820x640")  # 宽+20, 高+40（含控件）
        self._roi_window.transient(self.root)
        self._roi_window.grab_set()
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.ico")
        if os.path.exists(icon_path):
            try:
                self._roi_window.iconbitmap(icon_path)
            except Exception as e:
                self.log_message(f"[WARN] ROI窗口图标加载失败: {e}")
        # Canvas 布局：可伸缩
        self._roi_canvas = ttk.Canvas(self._roi_window, bg="black")
        self._roi_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self._roi_window.grid_rowconfigure(0, weight=1)
        self._roi_window.grid_columnconfigure(0, weight=1)

        # 显示初始图像
        self._roi_photo = ImageTk.PhotoImage(Image.fromarray(self._roi_display_rgb))
        self._canvas_img_id = self._roi_canvas.create_image(0, 0, anchor=NW, image=self._roi_photo)

        # 初始化 ROI 点
        self._roi_points_scaled = []

        # 绑定事件
        self._roi_canvas.bind("<Button-1>", self._on_roi_click)
        self._roi_canvas.bind("<Button-3>", self._on_roi_right_click)
        self._roi_window.bind("<Configure>", self._on_roi_window_resize)

        # 按钮区域
        btn_frame = ttk.Frame(self._roi_window)
        btn_frame.grid(row=1, column=0, pady=5)
        ttk.Label(self._roi_window, text="左键添加点，右键删除最后一点,点完成【确认】设置关注区域，点【取消】或直接关闭窗口取消设置。", foreground="gray").grid(row=2, column=0)
        
        ttk.Button(btn_frame, text="确认", bootstyle=SUCCESS, command=self._finish_roi_selection).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="重置", bootstyle=WARNING, command=self._reset_roi_points).pack(side=LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", bootstyle=SECONDARY, command=self._roi_window.destroy).pack(side=LEFT, padx=5)

        self.log_message("请在弹出窗口中绘制关注区域（至少3个点）")


    #ROI选择窗口核心缩放逻辑
    def _resize_roi_display(self, canvas_w, canvas_h):
        """根据 Canvas 尺寸，等比缩放原始帧"""
        if not hasattr(self, '_roi_frame_orig'):
            return

        frame = self._roi_frame_orig
        h, w = self._roi_orig_h, self._roi_orig_w

        # 计算缩放比例（保持宽高比）
        scale_w = canvas_w / w
        scale_h = canvas_h / h
        scale = min(scale_w, scale_h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放图像
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self._roi_display_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._roi_current_scale = scale  # 用于坐标映射
        self._roi_canvas_w = new_w
        self._roi_canvas_h = new_h
        self._roi_canvas_offset_x = (canvas_w - new_w) // 2
        self._roi_canvas_offset_y = (canvas_h - new_h) // 2

    #窗口调整事件处理
    def _on_roi_window_resize(self, event):
        # 忽略按钮区域 resize（只响应 Canvas 区域变化）
        if event.widget != self._roi_window:
            return
        # 延迟处理，避免频繁触发
        if hasattr(self, '_roi_resize_after_id'):
            self._roi_window.after_cancel(self._roi_resize_after_id)
        self._roi_resize_after_id = self._roi_window.after(100, self._perform_roi_resize, event.width, event.height - 100)  # 扣除按钮高度

    def _perform_roi_resize(self, win_w, win_h):
        if win_w <= 1 or win_h <= 1:
            return
        # 重新缩放图像
        self._resize_roi_display(win_w, win_h)
        # 更新 Canvas 显示
        self._roi_photo = ImageTk.PhotoImage(Image.fromarray(self._roi_display_rgb))
        self._roi_canvas.itemconfig(self._canvas_img_id, image=self._roi_photo)
        self._roi_canvas.config(width=win_w, height=win_h)
        # 移动图像到居中位置
        self._roi_canvas.coords(self._canvas_img_id, self._roi_canvas_offset_x, self._roi_canvas_offset_y)
        # 重绘 ROI 点（需重新计算位置）
        self._redraw_roi()

    #辅助方法：鼠标交互
    def cancel_roi(self):
        self.roi = None
        self.roi_label.config(text="关注区域未设置", foreground="gray")
        self.roi_button.config(text="选取关注区域", command=self.select_roi)
        self.log_message("已取消关注区域")

    def _on_roi_click(self, event):
        offset_x = getattr(self, '_roi_canvas_offset_x', 0)
        offset_y = getattr(self, '_roi_canvas_offset_y', 0)
        # 转换为图像坐标系（扣除居中偏移）
        img_x = event.x - offset_x
        img_y = event.y - offset_y
        # 检查是否在图像范围内
        if 0 <= img_x < self._roi_canvas_w and 0 <= img_y < self._roi_canvas_h:
            self._roi_points_scaled.append((img_x, img_y))
            self._redraw_roi()

    def _on_roi_right_click(self, event):
        if self._roi_points_scaled:
            self._roi_points_scaled.pop()
            self._redraw_roi()

    def _redraw_roi(self):
        self._roi_canvas.delete("roi_line", "roi_point")
        if not hasattr(self, '_roi_points_scaled') or not self._roi_points_scaled:
            return

        offset_x = getattr(self, '_roi_canvas_offset_x', 0)
        offset_y = getattr(self, '_roi_canvas_offset_y', 0)

        # 重绘点
        for x, y in self._roi_points_scaled:
            self._roi_canvas.create_oval(
                x + offset_x - 3, y + offset_y - 3,
                x + offset_x + 3, y + offset_y + 3,
                fill="red", tags="roi_point"
            )

        # 重绘连线
        n = len(self._roi_points_scaled)
        if n > 1:
            for i in range(n - 1):
                x1, y1 = self._roi_points_scaled[i]
                x2, y2 = self._roi_points_scaled[i + 1]
                self._roi_canvas.create_line(
                    x1 + offset_x, y1 + offset_y,
                    x2 + offset_x, y2 + offset_y,
                    fill="lime", width=2, tags="roi_line"
                )
        if n >= 3:
            x1, y1 = self._roi_points_scaled[-1]
            x2, y2 = self._roi_points_scaled[0]
            self._roi_canvas.create_line(
                x1 + offset_x, y1 + offset_y,
                x2 + offset_x, y2 + offset_y,
                fill="lime", width=2, dash=(4, 2), tags="roi_line"
            )

    #完成 ROI 选择
    def _finish_roi_selection(self):
        if len(self._roi_points_scaled) < 3:
            Messagebox.showwarning("ROI 无效", "请至少选择 3 个点")
            return

        # 从显示坐标 → 原始视频坐标
        roi_points_orig = []
        scale = self._roi_current_scale
        for x_scaled, y_scaled in self._roi_points_scaled:
            x_orig = int(round(x_scaled / scale))
            y_orig = int(round(y_scaled / scale))
            roi_points_orig.append((x_orig, y_orig))

        # ===== 合法性校验 =====
        h, w = self._roi_orig_h, self._roi_orig_w
        roi_array = np.array(roi_points_orig, dtype=np.int32)

        if roi_array.ndim != 2 or roi_array.shape[1] != 2:
            self.log_message("ROI 格式错误")
            return

        xs, ys = roi_array[:, 0], roi_array[:, 1]
        if not (np.all(xs >= 0) and np.all(xs < w) and np.all(ys >= 0) and np.all(ys < h)):
            self.log_message(f"ROI 点超出视频帧范围（宽{w}, 高{h}）")
            return

        area = cv2.contourArea(roi_array)
        if area <= 1.0:
            self.log_message("ROI 区域无效：面积过小或退化")
            return

        # 保存
        self.roi = roi_array.copy()
        self.roi_label.config(text="关注区域已设置", foreground="orange")
        self.roi_button.config(text="取消选取", command=self.cancel_roi)
        self.log_message(f"成功设置 ROI，面积: {area:.1f} 像素")

        self._roi_window.destroy()

    #重置 & 取消ROI    
    def _reset_roi_points(self):
        self._roi_points_scaled.clear()
        self._redraw_roi()

    def cancel_roi(self):
        self.roi = None
        self.roi_label.config(text="关注区域未设置", foreground="gray")
        self.roi_button.config(text="选取关注区域", command=self.select_roi)
        self.log_message("已取消关注区域")

    #ROI-------------

    # 同步预览实现 
    def toggle_preview(self):
        """切换同步预览（支持多文件连续预览）"""
        if not self.file_paths:
            self.log_message("请先选择视频文件")
            return
            
        if self._preview_active:
            # 关闭预览：清空预览标签，终止更新循环
            self._preview_label.config(image="", text="预览已关闭")
            self._preview_active = False
            self.preview_button.config(text="开启实时预览")
            self.preview_status.config(text="预览: 关闭", foreground="gray")
            self.log_message("实时预览已关闭")
        else:
            # 开启预览：启动更新循环，无需等待单个文件开始
            self._preview_active = True
            self.preview_button.config(text="关闭实时预览")
            self.preview_status.config(text="预览: 开启", foreground="orange")
            self.log_message("实时预览已开启，将跟随多文件处理连续同步显示")
            # 立即启动预览更新循环
            self._update_preview_display()

    def _start_preview(self):
        """启动同步预览"""
        self._preview_active = True
        
        self._preview_window = ttk.Toplevel(self.root)
        self._preview_window.title("实时预览")
        self._preview_window.geometry("800x600")
        self._preview_window.protocol("WM_DELETE_WINDOW", self._close_preview)
        
        self._preview_label = ttk.Label(self._preview_window)
        self._preview_label.pack(fill=BOTH, expand=True)
        
        self.preview_button.config(text="关闭实时预览")
        self.preview_status.config(text="预览: 开启", foreground="orange")
        
        self._update_preview_display()

    def _update_preview_display(self):
        if not self._preview_active:
            return

        frame_processed = False
        try:
            item = self._preview_queue.get_nowait()
            if isinstance(item, (tuple, list)) and len(item) == 3:
                frame, frame_idx, annotate_info = item
                if frame is not None:
                    # 绘制ROI
                    if self.roi is not None:
                        cv2.polylines(frame, [self.roi], isClosed=True, color=(0, 255, 0), thickness=2)

                    # 绘制标注
                    for (x1, y1, x2, y2, class_display, score) in annotate_info:
                        color = COLORS[hash(class_display) % len(COLORS)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        text = f"{class_display} {score:.2f}"
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        text_x = x1
                        text_y = y1 - 10 if y1 - 10 > 0 else y1 + text_size[1] + 10
                        cv2.rectangle(
                            frame,
                            (text_x, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 2, text_y + 2),
                            color,
                            -1
                        )
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # 转换并显示
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    preview_width = self._preview_label.winfo_width() or 600
                    preview_height = self._preview_label.winfo_height() or 400
                    pil_image.thumbnail((preview_width, preview_height), Image.LANCZOS)
                    self._current_preview_photo = ImageTk.PhotoImage(pil_image)
                    self._preview_label.config(image=self._current_preview_photo, text="")
                    self._preview_label.image = self._current_preview_photo
                    frame_processed = True

        except queue.Empty:
            pass
        except Exception as e:
            self.log_message(f"[预览错误] {e}")

        # 无论是否处理帧，只要 _preview_active，就继续递归！
        if self._preview_active:
            self.root.after(30, self._update_preview_display)

    def _close_preview(self):
        """关闭预览（嵌入模式，仅清空标签）"""
        self._preview_active = False
        
        # 清空预览标签
        self._preview_label.config(image="")
        # 清空预览队列
        while not self._preview_queue.empty():
            try:
                self._preview_queue.get_nowait()
            except:
                break
        
        # 更新按钮和状态
        self.preview_button.config(text="开启实时预览")
        self.preview_status.config(text="预览: 关闭", foreground="gray")

    def save_image_safely(self, image, save_path):
        try:
            self.save_queue.put_nowait((image, save_path))
            return f"截图: {save_path}"
        except queue.Full:
            return f"截图队列已满，跳过: {save_path}"

    #  ROI 检查
    def _is_point_in_roi(self, x, y):
        if self.roi is None:
            return True
        try:
            # 确保 ROI 是有效 np.int32 (N, 2)
            if not isinstance(self.roi, np.ndarray) or self.roi.shape[1] != 2:
                return True  # 退化为无 ROI
            pt = (int(round(x)), int(round(y)))
            result = cv2.pointPolygonTest(self.roi, pt, measureDist=False)
            return result >= 0
        except Exception as e:
            self.log_message(f"[ROI 安全防护] 检查点 ({x:.1f},{y:.1f}) 时出错: {e}")
            return True  # 安全默认：放行

    # 核心：轻量移动检测 
    def _box_iou_simple(self, box1, box2):
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

    def _is_target_moving_enhanced(self, object_key, current_box, current_time, prev_positions, stable_since, movement_buffer):
        prev_box = prev_positions.get(object_key)
        if prev_box is None:
            return True

        iou = self._box_iou_simple(prev_box, current_box)
        if iou > self.MOVEMENT_IOU_THRESHOLD:
            if object_key not in stable_since:
                stable_since[object_key] = current_time
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

        moved = rel_dx > self.MOVEMENT_RELATIVE_THRESHOLD or rel_dy > self.MOVEMENT_RELATIVE_THRESHOLD

        if object_key not in movement_buffer:
            movement_buffer[object_key] = 0

        if moved:
            movement_buffer[object_key] += 1
        else:
            movement_buffer[object_key] = 0

        last_stable = stable_since.get(object_key, -10)
        recently_stable = (current_time - last_stable) < self.MOVEMENT_STABLE_SECONDS
        if recently_stable and movement_buffer[object_key] >= self.MOVEMENT_CONSECUTIVE_FRAMES:
            del stable_since[object_key]
            return True
        elif not recently_stable:
            return movement_buffer[object_key] >= self.MOVEMENT_CONSECUTIVE_FRAMES
        return False

    # ==============================
    def read_frames_producer(self, video_path, skip_frames):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        with self.progress_lock:
            self.total_frames = total_frames

        while cap.isOpened() and not self.stopped:
            while self.paused:
                time.sleep(0.01)
                if self.stopped:
                    break
            if self.stopped:
                break

            ret = cap.grab()
            if not ret:
                break

            if frame_idx % skip_frames == 0:
                ret, frame = cap.retrieve()
                if not ret:
                    break

                with self.progress_lock:
                    self.current_frame = frame_idx

                resized_frame = cv2.resize(frame, (self.INFERENCE_SIZE, self.INFERENCE_SIZE))
                try:
                    self.frame_queue.put((frame.copy(), resized_frame, frame_idx), timeout=1.0)
                except queue.Full:
                    time.sleep(0.01)

            frame_idx += 1

        cap.release()
        self.frame_queue.put(None)
        # 预览队列放入结束标识
        if self._preview_active:
            try:
                self._preview_queue.put((None, -1, []), timeout=0.5)
            except queue.Full:
                self.log_message("[WARN] 预览队列满，无法发送视频结束信号")

    #进度更新：带暂停检查
    def _progress_updater(self, file_idx, total_files):
        """独立线程：定期更新进度条（带暂停检查）"""
        last_frame = -1
        while not self.stopped:
            # 暂停时跳过进度更新
            if self.paused_display:
                time.sleep(0.1)
                continue
                
            with self.progress_lock:
                current = self.current_frame
                total = self.total_frames
                current_file_idx = self._current_file_idx

            if current_file_idx != file_idx:
                break

            if total > 0 and current != last_frame:
                progress = min(99, int(100 * current / total))
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda f=file_idx+1, t=total_files, p=progress: 
                    self.status_label.config(text=f"处理文件 {f}/{t} | 进度: {p}%"))
                last_frame = current

            time.sleep(0.1)

        if not self.stopped and file_idx == total_files - 1:
            self.root.after(0, lambda: self.progress_var.set(100))

    def process_videos(self, file_paths: list, save_folder: str, selected_classes: list, skip_frames: int, only_moving: bool,
                    annotate_objects: bool, confidence_threshold: float):
        try:
            total_files = len(file_paths)
            for file_idx, file_path in enumerate(file_paths):
                if self.stopped:
                    break
                with self.progress_lock:
                    self._current_file_idx = file_idx
                    self.current_frame = 0
                    self.total_frames = 0

                self.log_message(f"切换到第 {file_idx+1}/{total_files} 个文件，预览已同步跟随")
                self.roi_entered_boxes = []
                prev_positions = {}
                movement_buffer = {}
                stable_since = {}
                self.frame_queue = queue.Queue(maxsize=10)
                cap_temp = cv2.VideoCapture(file_path)
                orig_h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                orig_w = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap_temp.release()
                reader_thread = threading.Thread(
                    target=self.read_frames_producer,
                    args=(file_path, skip_frames),
                    daemon=True
                )
                reader_thread.start()
                progress_thread = threading.Thread(
                    target=self._progress_updater,
                    args=(file_idx, total_files),
                    daemon=True
                )
                progress_thread.start()
                batch_original_frames = []
                batch_input_frames = []
                batch_frame_counts = []
                last_inference_time = time.time()
                while True:
                    if self.stopped:
                        break
                    try:
                        item = self.frame_queue.get(timeout=2)
                        if item is None:
                            time.sleep(0.1)
                            break
                    except queue.Empty:
                        continue
                    orig_frame, input_frame, frame_count = item
                    batch_original_frames.append(orig_frame)
                    batch_input_frames.append(input_frame)
                    batch_frame_counts.append(frame_count)
                    if len(batch_input_frames) >= self.BATCH_SIZE:
                        self._run_batch_inference_optimized(
                            batch_input_frames, batch_original_frames, batch_frame_counts,
                            selected_classes, only_moving, annotate_objects,
                            confidence_threshold, orig_w, orig_h,
                            save_folder, file_path,
                            prev_positions, stable_since, movement_buffer
                        )
                        current_time = time.time()
                        if current_time > last_inference_time:
                            self.stats['fps'] = len(batch_frame_counts) / (current_time - last_inference_time)
                            last_inference_time = current_time
                        batch_original_frames.clear()
                        batch_input_frames.clear()
                        batch_frame_counts.clear()
                if batch_input_frames:
                    self._run_batch_inference_optimized(
                        batch_input_frames, batch_original_frames, batch_frame_counts,
                        selected_classes, only_moving, annotate_objects,
                        confidence_threshold, orig_w, orig_h,
                        save_folder, file_path,
                        prev_positions, stable_since, movement_buffer
                    )
                progress_thread.join(timeout=1)
                reader_thread.join(timeout=2)

            if not self.stopped:
                self.update_status("所有文件处理完成！")

        except Exception as e:
            self.log_message(f"处理出错: {e}\n{traceback.format_exc()}")
            self.update_status(f"错误: {str(e)}")
        finally:
            def reset_ui():
                self.start_button.config(state=NORMAL)
                self.pause_button.config(state=DISABLED)
                self.stop_button.config(state=DISABLED)
                if self.stopped:
                    self.update_status("处理已停止")
                    self.log_message("处理已停止")
                    if self._preview_active:
                        self._preview_label.config(text="处理已停止，预览中断", image="")
                else:
                    self.update_status("所有文件处理完成！")
            self.root.after_idle(reset_ui)

            # 所有任务结束后，清空预览队列（可保留）
            while not self._preview_queue.empty():
                try:
                    self._preview_queue.get_nowait()
                except:
                    break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    #---------------------------------------------------------
    def _run_batch_inference_optimized(self, batch_input_frames, batch_original_frames, batch_frame_counts,
                                    selected_classes, only_moving, annotate_objects,
                                    confidence_threshold, orig_w, orig_h,
                                    save_folder, file_path,
                                    prev_positions, stable_since, movement_buffer):
        try:
            if CUDA_STREAM is not None:
                with torch.cuda.stream(CUDA_STREAM):
                    results_list = self.model(
                        batch_input_frames,
                        conf=confidence_threshold,
                        iou=self.NMS_IOU_THRESHOLD,
                        verbose=False
                    )
                torch.cuda.current_stream().synchronize()
            else:
                results_list = self.model(
                    batch_input_frames,
                    conf=confidence_threshold,
                    iou=self.NMS_IOU_THRESHOLD,
                    verbose=False
                )

            for idx, results in enumerate(results_list):
                frame = batch_original_frames[idx]
                frame_count = batch_frame_counts[idx]
                if not (results and len(results.boxes) > 0):
                    # 即使无检测，也推送空标注帧
                    if self._preview_active:
                        try:
                            if self._preview_queue.full():
                                self._preview_queue.get_nowait()
                            self._preview_queue.put_nowait((frame.copy(), frame_count, []))
                        except queue.Full:
                            pass
                    continue

                # 提取原始检测结果
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)

                # 缩放回原始尺寸
                scale_x = orig_w / self.INFERENCE_SIZE
                scale_y = orig_h / self.INFERENCE_SIZE
                boxes_original = boxes.copy()
                boxes_original[:, [0, 2]] *= scale_x
                boxes_original[:, [1, 3]] *= scale_y
                boxes_original = np.clip(boxes_original, 0, [orig_w, orig_h, orig_w, orig_h])

                # 过滤未勾选类别
                filtered_boxes = []
                filtered_scores = []
                filtered_class_ids = []
                for box, score, cls_id in zip(boxes_original, scores, class_ids):
                    if cls_id >= len(COCO_CLASSES):
                        continue
                    class_name = COCO_CLASSES[cls_id]
                    if class_name not in selected_classes:
                        continue
                    filtered_boxes.append(box)
                    filtered_scores.append(score)
                    filtered_class_ids.append(cls_id)
                if not filtered_boxes:
                    if self._preview_active:
                        try:
                            if self._preview_queue.full():
                                self._preview_queue.get_nowait()
                            self._preview_queue.put_nowait((frame.copy(), frame_count, []))
                        except queue.Full:
                            pass
                    continue

                #跨类别高 IoU 去重 
                detections = [(filtered_scores[i], filtered_boxes[i], filtered_class_ids[i]) for i in range(len(filtered_scores))]
                detections.sort(key=lambda x: x[0], reverse=True)
                to_keep = [True] * len(detections)
                for i in range(len(detections)):
                    if not to_keep[i]:
                        continue
                    score_i, box_i, cls_i = detections[i]
                    for j in range(i + 1, len(detections)):
                        if not to_keep[j]:
                            continue
                        score_j, box_j, cls_j = detections[j]
                        if cls_i == cls_j:
                            continue
                        x1, y1, x2, y2 = box_i
                        x1_p, y1_p, x2_p, y2_p = box_j
                        inter_x1 = max(x1, x1_p)
                        inter_y1 = max(y1, y1_p)
                        inter_x2 = min(x2, x2_p)
                        inter_y2 = min(y2, y2_p)
                        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                            area1 = (x2 - x1) * (y2 - y1)
                            area2 = (x2_p - x1_p) * (y2_p - y1_p)
                            union_area = area1 + area2 - inter_area
                            iou = inter_area / union_area if union_area > 0 else 0.0
                            if iou > 0.85:
                                to_keep[j] = False

                kept_detections = [detections[i] for i in range(len(detections)) if to_keep[i]]
                if not kept_detections:
                    if self._preview_active:
                        try:
                            if self._preview_queue.full():
                                self._preview_queue.get_nowait()
                            self._preview_queue.put_nowait((frame.copy(), frame_count, []))
                        except queue.Full:
                            pass
                    continue
                kept_scores, kept_boxes, kept_class_ids = zip(*kept_detections)
                kept_scores = np.array(kept_scores)
                kept_boxes = np.array(kept_boxes)
                kept_class_ids = np.array(kept_class_ids)

                # 构造预览标注信息（仅 ROI 内目标）
                preview_annotate_info = []
                for box, score, cls_id in zip(kept_boxes, kept_scores, kept_class_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    if not self._is_point_in_roi((x1 + x2) / 2.0, (y1 + y2) / 2.0):
                        continue
                    class_name = COCO_CLASSES[cls_id]
                    class_display = COCO_CLASS_MAPPING.get(class_name, class_name)
                    preview_annotate_info.append((x1, y1, x2, y2, class_name, float(score)))

                # 将帧 + 标注 一次性入队（不再分离）
                if self._preview_active:
                    try:
                        if self._preview_queue.full():
                            self._preview_queue.get_nowait()
                        # 现在队列中每个 item 是 (frame, frame_idx, annotate_info)
                        self._preview_queue.put_nowait((frame.copy(), frame_count, preview_annotate_info))
                    except queue.Full:
                        pass

                # 截图逻辑 
                for box, score, cls_id in zip(kept_boxes, kept_scores, kept_class_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    current_box = (x1, y1, x2, y2)
                    if not self._is_point_in_roi((x1 + x2) / 2.0, (y1 + y2) / 2.0):
                        continue

                    is_first_appearance = True
                    for recorded_box, recorded_cls in self.roi_entered_boxes:
                        if recorded_cls == cls_id and self._box_iou_simple(current_box, recorded_box) > self.ROI_IOU_THRESHOLD:
                            is_first_appearance = False
                            break

                    should_process = False
                    if only_moving:
                        if is_first_appearance:
                            should_process = True
                        else:
                            obj_key = f"{cls_id}_{x1}_{y1}_{x2}_{y2}"
                            if self._is_target_moving_enhanced(obj_key, current_box, frame_count / 30,
                                                            prev_positions, stable_since, movement_buffer):
                                should_process = True
                            prev_positions[obj_key] = current_box
                    else:
                        should_process = True

                    if not should_process:
                        continue

                    if is_first_appearance:
                        self.roi_entered_boxes.append((current_box, cls_id))

                    class_name = COCO_CLASSES[cls_id]
                    class_display = COCO_CLASS_MAPPING.get(class_name, class_name)
                    self.stats['total_targets'] += 1
                    self.stats['targets_by_class'][class_display] = self.stats['targets_by_class'].get(class_display, 0) + 1

                    annotate_frame = frame.copy() if annotate_objects else frame
                    if annotate_objects:
                        color = COLORS[hash(class_name) % len(COLORS)]
                        cv2.rectangle(annotate_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotate_frame, f"{class_name} {score:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    safe_base = sanitize_filename(os.path.splitext(os.path.basename(file_path))[0])
                    safe_class = sanitize_filename(class_display)
                    save_path = os.path.join(save_folder, f"{safe_base}_frame{frame_count}_{safe_class}.jpg")
                    msg = self.save_image_safely(annotate_frame, save_path)
                    self.log_message(f"发现目标: {class_display} 置信度{score:.2f} → {msg}")

        except Exception as e:
            self.log_message(f"批量推理出错: {e}")
            import traceback
            self.log_message(traceback.format_exc())

    #---------------------------------------------------------
    def start_processing(self):
        if not self.file_paths:
            self.log_message("错误: 未选择视频文件")
            return
        if not self.save_folder:
            self.log_message("错误: 未选择保存路径")
            return
        try:
            with open(os.path.join(self.save_folder, "test.tmp"), 'w') as _:
                pass
            os.remove(os.path.join(self.save_folder, "test.tmp"))
        except Exception as e:
            Messagebox.showerror("路径错误", f"保存路径不可写: {e}")
            return

        selected = [self.class_names[i] for i, v in enumerate(self.class_vars) if v.get()]
        if not selected:
            self.log_message("错误: 未选择任何目标类")
            return

        try:
            skip_frames = max(1, min(64, int(self.speed_entry.get())))
        except ValueError:
            skip_frames = 1
        try:
            conf = max(0.0, min(1.0, float(self.confidence_entry.get())))
        except ValueError:
            conf = 0.1

        only_moving = self.only_moving_var.get() == 1
        annotate = self.annotate_var.get() == 1

        self._auto_adjust_batch_size()

        self.paused = False
        self.stopped = False
        # 重置暂停显示标志
        self.paused_display = False
        self.start_button.config(state=DISABLED)
        self.pause_button.config(state=NORMAL)
        self.stop_button.config(state=NORMAL)
        self.log_message("开始处理...")

        threading.Thread(target=self.process_videos, args=(
            self.file_paths, self.save_folder, selected, skip_frames, only_moving, annotate, conf
        ), daemon=True).start()

    def pause_processing(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="继续处理")
            self.log_message("处理已暂停")
            # 设置暂停显示标志
            self.paused_display = True
            self.update_status("处理已暂停...")
        else:
            self.pause_button.config(text="暂停处理")
            self.log_message("处理已继续")
            # 清除暂停显示标志
            self.paused_display = False
            if self.current_processing_status:
                self.update_status(self.current_processing_status)
            else:
                self.update_status("处理中...")

    def stop_processing(self):
        if not self.stopped:
            self.stopped = True
            self.paused = False
            self.paused_display = False 
            self.current_processing_status = ""
            self.log_message("正在停止处理...")
            self.update_status("正在停止...")
        while not self.save_queue.empty():
            try:
                self.save_queue.get_nowait()
                self.save_queue.task_done()
            except:
                break

    def select_all_classes(self):
        for v in self.class_vars:
            v.set(1)
        self.log_message("全选所有类别")

    def deselect_all_classes(self):
        for v in self.class_vars:
            v.set(0)
        self.log_message("取消已选目标")

    def update_stats_periodically(self):
        """定期更新统计面板"""
        if self._ui_ready:
            self.fps_label.config(text=f"FPS(每秒处理帧数): {self.stats['fps']:.1f}")
            self.targets_label.config(text=f"目标截图总数: {self.stats['total_targets']}")
            if self.gpu_handle is not None:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_mem_gb = mem_info.used / (1024**3)
                    self.gpu_label.config(text=f"占用 GPU 显存: {gpu_mem_gb:.1f} GB")
                except:
                    self.gpu_label.config(text="占用 GPU 显存: 错误")
            else:
                self.gpu_label.config(text="占用 GPU 显存: N/A")
        self.root.after(500, self.update_stats_periodically)

    def __del__(self):
        if hasattr(self, 'save_queue'):
            self.save_queue.put(None)
        if PYNVML_AVAILABLE and hasattr(self, 'gpu_handle'):
            try:
                pynvml.nvmlShutdown()
            except:
                pass


if __name__ == "__main__":
    # 初始化ttkbootstrap窗口，可选择主题（如darkly、yeti、simplex、cyborg、superhero等）
    root = ttk.Window(themename="darkly")
    app = VideoDetectionApp(root)
    root.mainloop()
