# main.py
#update 2026-03-04
import sys
import os
import cv2
import time
import threading
import queue
import numpy as np
import configparser
import traceback
import datetime
import unicodedata
import re
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QCheckBox, QProgressBar, QTextEdit, QFileDialog, QMessageBox,
    QScrollArea, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QDialog,
    QSizePolicy, QSplashScreen
)
from PySide6.QtGui import QPixmap, QImage, QIcon, QPen, QColor, QPainter, QFont
from PySide6.QtCore import Qt, QTimer, QPoint, Signal, QRect, Slot

# 尝试导入 GPU 监控库
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# --- 常量定义 ---
DEFAULT_MODEL_PATH = './models/yolo26x.pt'
DEFAULT_INFERENCE_SIZE = 640
DEFAULT_BATCH_SIZE = 8
DEFAULT_NMS_IOU_THRESHOLD = 0.45
DEFAULT_ROI_IOU_THRESHOLD = 0.2
DEFAULT_MOVEMENT_IOU_THRESHOLD = 0.6
DEFAULT_MOVEMENT_RELATIVE_THRESHOLD = 0.02
DEFAULT_MOVEMENT_CONSECUTIVE_FRAMES = 5
DEFAULT_MOVEMENT_STABLE_SECONDS = 5.0
CONFIG_FILE_NAME = 'set.ini'
LOG_DIR_NAME = 'logs'
LOG_FILE_NAME = 'app_runtime.log'
VIEWER_EXE_NAME = 'viewer.exe'

# COCO 类别映射
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

# 全局状态
ULTRALYTICS_AVAILABLE: Optional[bool] = None

# --- 工具函数 ---
def get_base_path():
    """获取程序基础路径，兼容打包环境"""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent

def sanitize_filename(filename: str) -> str:
    """清理文件名中的非法字符"""
    cleaned = "".join(c for c in filename if unicodedata.category(c) not in ("Cc", "Cf") and c not in r'<>:"/\|?*')
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned[:100] or "unnamed"

def cv2_to_qimage(cv_img: np.ndarray) -> QImage:
    """将 OpenCV 图像转换为 Qt 图像"""
    if cv_img is None or cv_img.size == 0:
        return QImage()
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

# --- 自定义控件 ---
class ClickableLabel(QLabel):
    """支持点击事件的标签"""
    clicked = Signal(QPoint)
    right_clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.position().toPoint())
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit()

class ROIDialog(QDialog):
    """ROI 区域选择对话框"""
    def __init__(self, parent: QWidget, frame: np.ndarray):
        super().__init__(parent)
        self.setWindowTitle("选择关注区域 (ROI)")
        self.resize(800, 600)
        self.frame = frame.copy()
        self.h, self.w = self.frame.shape[:2]
        self.points_orig: List[Tuple[float, float]] = []
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._painting = False

        self.image_label = ClickableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.clicked.connect(self._on_image_click)
        self.image_label.right_clicked.connect(self._on_image_right_click)
        self.image_label.setMinimumSize(0, 0)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(400, 300)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)

        btn_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("确认")
        self.reset_btn = QPushButton("重置")
        self.cancel_btn = QPushButton("取消")
        self.confirm_btn.clicked.connect(self.accept)
        self.reset_btn.clicked.connect(self.reset)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.confirm_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.update_display()

    def update_display(self):
        if self._painting:
            return
        self._painting = True
        try:
            label_rect = self.image_label.contentsRect()
            label_w, label_h = label_rect.width(), label_rect.height()
            if label_w <= 1 or label_h <= 1:
                return
            scale_w = label_w / self.w
            scale_h = label_h / self.h
            self.scale = min(scale_w, scale_h)
            new_w = int(self.w * self.scale)
            new_h = int(self.h * self.scale)
            resized = cv2.resize(self.frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            qimg = cv2_to_qimage(resized)
            self.pixmap = QPixmap.fromImage(qimg)
            self.offset_x = (label_w - new_w) // 2
            self.offset_y = (label_h - new_h) // 2

            canvas = QPixmap(label_w, label_h)
            canvas.fill(Qt.black)
            painter = QPainter(canvas)
            painter.drawPixmap(self.offset_x, self.offset_y, self.pixmap)
            painter.end()
            self.image_label.setPixmap(canvas)
            self.redraw_roi()
        finally:
            self._painting = False

    def redraw_roi(self):
        if self._painting or not hasattr(self, 'pixmap') or self.pixmap.isNull():
            return
        self._painting = True
        try:
            label_rect = self.image_label.contentsRect()
            canvas = QPixmap(label_rect.width(), label_rect.height())
            canvas.fill(Qt.black)
            painter = QPainter(canvas)
            painter.drawPixmap(self.offset_x, self.offset_y, self.pixmap)

            if len(self.points_orig) >= 1:
                points_on_label = [
                    QPoint(int(x * self.scale + self.offset_x), int(y * self.scale + self.offset_y))
                    for x, y in self.points_orig
                ]
                pen = QPen(QColor("lime"), 2)
                painter.setPen(pen)
                for i in range(len(points_on_label) - 1):
                    painter.drawLine(points_on_label[i], points_on_label[i + 1])
                if len(points_on_label) >= 3:
                    pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(points_on_label[-1], points_on_label[0])

                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor("red"))
                for pt in points_on_label:
                    painter.drawEllipse(pt, 4, 4)
            painter.end()
            self.image_label.setPixmap(canvas)
        finally:
            self._painting = False

    def _on_image_click(self, pos: QPoint):
        x_label = pos.x()
        y_label = pos.y()
        x_img = (x_label - self.offset_x) / self.scale
        y_img = (y_label - self.offset_y) / self.scale
        if 0 <= x_img < self.w and 0 <= y_img < self.h:
            self.points_orig.append((x_img, y_img))
            self.redraw_roi()

    def _on_image_right_click(self):
        if self.points_orig:
            self.points_orig.pop()
            self.redraw_roi()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(50, self.update_display)

    def reset(self):
        self.points_orig.clear()
        self.redraw_roi()

    def get_roi(self) -> Optional[np.ndarray]:
        if len(self.points_orig) < 3:
            return None
        return np.array([(int(round(x)), int(round(y))) for x, y in self.points_orig], dtype=np.int32)

# --- 主窗口 ---
class MainWindow(QMainWindow):
    # 线程安全信号
    log_signal = Signal(str)
    status_signal = Signal(str)
    button_state_signal = Signal(bool, bool, bool)  # start, pause, stop
    processing_finished_signal = Signal(str)
    model_loaded_signal = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频目标检测截图工具 V3.0 by geckotao")
        self.resize(1200, 800)

        # 图标设置（兼容打包）
        icon_path = get_base_path() / "favicon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # 数据初始化
        self.class_names = {i: name for i, name in enumerate(COCO_CLASSES)}
        self.CHINESE_TO_ENGLISH = {v: k for k, v in COCO_CLASS_MAPPING.items()}
        self.class_checkboxes: List[QCheckBox] = []

        # 日志路径初始化（兼容打包）
        self.log_dir_path = get_base_path() / LOG_DIR_NAME
        self.log_dir_path.mkdir(exist_ok=True)
        self.log_file_path = self.log_dir_path / LOG_FILE_NAME
        
        # 清空旧日志
        try:
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                pass
        except Exception:
            pass

        # 状态变量
        self.model = None
        self._model_loaded = False
        self._preview_active = False
        self.paused = False
        self.stopped = False
        self.file_paths: List[str] = []
        self.save_folder = ""
        self.roi: Optional[np.ndarray] = None
        self.current_frame = 0
        self.total_frames = 0
        self._current_file_idx = -1
        self._processing_finished_normally = False
        self._generated_screenshot_paths: List[str] = []

        # 统计信息
        self.stats = {
            'fps': 0.0,
            'total_targets': 0,
            'targets_by_class': {},
            'gpu_mem': 'N/A'
        }

        # 进度控制
        self._progress_current = 0
        self._progress_total = 1
        self._processing_file_index = -1
        self._total_files = 1

        # GPU 监控
        self.gpu_handle = None
        self._model_load_triggered = False
        self._nvml_initialized = False  
        self._nvml_error_logged = False  
        self.torch = None
        self.force_cpu_mode = False

        # 队列与线程
        self._preview_queue = queue.Queue(maxsize=10)
        self.save_queue = queue.Queue(maxsize=100)
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        # UI 初始化
        self.setup_ui()
        self.load_configuration()

        # 定时器初始化（但不启动）
        self.stats_timer = QTimer()
        self.stats_timer.setTimerType(Qt.PreciseTimer)
        self.stats_timer.timeout.connect(self.update_stats_periodically)

        self.preview_timer = QTimer()
        self.preview_timer.setTimerType(Qt.PreciseTimer)
        self.preview_timer.timeout.connect(self._update_preview_display)

        self.progress_timer = QTimer()
        self.progress_timer.setTimerType(Qt.PreciseTimer)
        self.progress_timer.timeout.connect(self._update_progress_bar)

        # 信号连接（使用 QueuedConnection 确保线程安全）
        self.model_loaded_signal.connect(self.on_model_loaded, Qt.QueuedConnection)
        self.processing_finished_signal.connect(self.on_processing_finished, Qt.QueuedConnection)
        self.log_signal.connect(self._update_log_ui, Qt.QueuedConnection)
        self.status_signal.connect(self._update_status_ui, Qt.QueuedConnection)
        self.button_state_signal.connect(self._update_button_state_ui, Qt.QueuedConnection)

        # 延迟启动定时器（确保事件循环已运行）
        QTimer.singleShot(100, self._start_timers)

    def _start_timers(self):
        """在所有 UI 初始化完成后启动定时器"""
        self.stats_timer.start(500)
        self.progress_timer.start(100)

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, '_model_load_triggered', False):
            self._model_load_triggered = True
            QTimer.singleShot(50, self.start_model_loading)

    @Slot()
    def start_model_loading(self):
        global ULTRALYTICS_AVAILABLE
        if ULTRALYTICS_AVAILABLE is False:
            self.update_status("环境错误：未安装 PyTorch 或 Ultralytics")
            self.button_state_signal.emit(False, False, False)
            return
        
        self.update_status("模型加载中...")
        self.button_state_signal.emit(False, False, False)
        self.log_message("开始加载模型...（首次启动约需 10-30 秒）")

        def _load_in_thread():
            success = self.load_model()
            self.model_loaded_signal.emit(success)

        threading.Thread(target=_load_in_thread, daemon=True).start()

    @Slot(bool)
    def on_model_loaded(self, success: bool):
        if success:
            self._model_loaded = True
            self.update_status("模型加载完成，可开始处理")
            self.button_state_signal.emit(True, False, False)
            self.log_message("程序初始化完成")
        else:
            self.update_status("模型加载失败，请检查配置或重启程序")
            QMessageBox.critical(self, "模型加载失败",
                "YOLO 模型加载失败，请检查：\n1. 模型文件路径是否正确\n2. CUDA 驱动是否兼容\n3. 显存是否充足")
            self.button_state_signal.emit(False, False, False)

    def setup_ui(self):
        """初始化界面布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧面板
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)
        self._setup_left_video_select(left_layout)
        self._setup_left_save_path(left_layout)
        self._setup_left_classes(left_layout)
        self._setup_left_options(left_layout)
        self._setup_left_controls(left_layout)
        self._setup_left_stats(left_layout)

        # 右侧面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(6)
        right_layout.setContentsMargins(8, 8, 8, 8)
        self.right_layout = right_layout
        self._setup_right_status(right_layout)
        self._setup_right_log(right_layout)
        self._setup_right_preview(right_layout)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 2)

    def _setup_left_video_select(self, layout: QVBoxLayout):
        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("选择视频文件:"))
        self.video_entry = QLineEdit()
        video_browse_btn = QPushButton("浏览")
        video_browse_btn.clicked.connect(self.select_video)
        video_layout.addWidget(self.video_entry)
        video_layout.addWidget(video_browse_btn)
        layout.addLayout(video_layout)

    def _setup_left_save_path(self, layout: QVBoxLayout):
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("截图保存路径:"))
        self.save_path_entry = QLineEdit()
        save_browse_btn = QPushButton("浏览")
        save_browse_btn.clicked.connect(self.select_save_path)
        save_layout.addWidget(self.save_path_entry)
        save_layout.addWidget(save_browse_btn)
        layout.addLayout(save_layout)

    def _setup_left_classes(self, layout: QVBoxLayout):
        classes_group = QGroupBox("选择目标类别")
        classes_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.select_all_classes)
        deselect_btn = QPushButton("取消已选")
        deselect_btn.clicked.connect(self.deselect_all_classes)
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(deselect_btn)
        classes_layout.addLayout(btn_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        grid_layout = QGridLayout(scroll_content)
        self.class_checkboxes = []
        default_selected = {'bicycle', 'car', 'motorcycle', 'bus', 'truck'}
        for i, (idx, name) in enumerate(self.class_names.items()):
            display = COCO_CLASS_MAPPING.get(name, name)
            cb = QCheckBox(display)
            cb.setChecked(name in default_selected)
            self.class_checkboxes.append(cb)
            row, col = divmod(i, 5)
            grid_layout.addWidget(cb, row, col)
        scroll_area.setWidget(scroll_content)
        classes_layout.addWidget(scroll_area)
        classes_group.setLayout(classes_layout)
        layout.addWidget(classes_group)

    def _setup_left_options(self, layout: QVBoxLayout):
        options_group = QGroupBox("处理选项")
        options_layout = QGridLayout()
        options_layout.addWidget(QLabel("跳帧数:"), 0, 0)
        self.speed_entry = QLineEdit("23")
        options_layout.addWidget(self.speed_entry, 0, 1)
        options_layout.addWidget(QLabel("(1=逐帧，值越大越快)"), 0, 2)
        options_layout.addWidget(QLabel("置信度阈值:"), 1, 0)
        self.confidence_entry = QLineEdit("0.1")
        options_layout.addWidget(self.confidence_entry, 1, 1)
        options_layout.addWidget(QLabel("(0.0-1.0)"), 1, 2)
        self.only_moving_var = QCheckBox("只对移动目标截图")
        self.only_moving_var.setChecked(True)
        self.annotate_var = QCheckBox("在截图中标注目标")
        options_layout.addWidget(self.only_moving_var, 2, 0)
        options_layout.addWidget(self.annotate_var, 2, 1)
        roi_layout = QHBoxLayout()
        self.roi_button = QPushButton("选取关注区域")
        self.roi_button.clicked.connect(self.select_roi)
        self.roi_label = QLabel("未选取关注区域")
        self.roi_label.setStyleSheet("color: gray;")
        roi_layout.addWidget(self.roi_button)
        roi_layout.addWidget(self.roi_label)
        options_layout.addLayout(roi_layout, 3, 0)
        preview_layout = QHBoxLayout()
        self.preview_button = QPushButton("开启实时预览")
        self.preview_button.clicked.connect(self.toggle_preview)
        self.preview_status = QLabel("预览：关闭")
        self.preview_status.setStyleSheet("color: gray;")
        preview_layout.addWidget(self.preview_button)
        preview_layout.addWidget(self.preview_status)
        options_layout.addLayout(preview_layout, 3, 1)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

    def _setup_left_controls(self, layout: QVBoxLayout):
        ctrl_layout = QHBoxLayout()
        self.start_button = QPushButton("开始处理")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_processing)
        self.pause_button = QPushButton("暂停处理")
        self.pause_button.clicked.connect(self.pause_processing)
        self.pause_button.setEnabled(False)
        self.stop_button = QPushButton("停止处理")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        ctrl_layout.addWidget(self.start_button)
        ctrl_layout.addWidget(self.pause_button)
        ctrl_layout.addWidget(self.stop_button)
        layout.addLayout(ctrl_layout)

    def _setup_left_stats(self, layout: QVBoxLayout):
        stats_group = QGroupBox("实时统计")
        stats_layout = QVBoxLayout()
        self.fps_label = QLabel("FPS(每秒处理帧数): 0.0")
        self.targets_label = QLabel("截图总数（未去重）: 0")
        self.gpu_label = QLabel("已用 GPU 显存：N/A")
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.targets_label)
        stats_layout.addWidget(self.gpu_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

    def _setup_right_status(self, layout: QVBoxLayout):
        self.status_label = QLabel("模型加载中...（首次启动约需 10-30 秒）")
        self.status_label.setStyleSheet("color: #666; font-weight: bold;")
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

    def _setup_right_log(self, layout: QVBoxLayout):
        log_group = QGroupBox("处理日志")
        log_group.setFixedHeight(150)
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

    def _setup_right_preview(self, layout: QVBoxLayout):
        preview_group = QGroupBox("实时预览")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("无实时预览画面")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(600, 400)
        self.preview_label.setStyleSheet("background-color: black; color: white;")
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group, stretch=1)

    # === 线程安全的 UI 更新槽函数 ===
    @Slot(str)
    def _update_log_ui(self, message: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        except Exception:
            pass
        if hasattr(self, 'log_text'):
            self.log_text.append(formatted)
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    @Slot(str)
    def _update_status_ui(self, text: str):
        if hasattr(self, 'status_label'):
            self.status_label.setText(text)

    @Slot(bool, bool, bool)
    def _update_button_state_ui(self, start_enabled: bool, pause_enabled: bool, stop_enabled: bool):
        if hasattr(self, 'start_button'):
            self.start_button.setEnabled(start_enabled)
        if hasattr(self, 'pause_button'):
            self.pause_button.setEnabled(pause_enabled)
        if hasattr(self, 'stop_button'):
            self.stop_button.setEnabled(stop_enabled)

    def log_message(self, message: str):
        """线程安全日志入口"""
        self.log_signal.emit(message)

    def update_status(self, text: str):
        """线程安全状态更新入口"""
        self.status_signal.emit(text)

    def _save_worker(self):
        """后台保存线程"""
        while True:
            try:
                task = self.save_queue.get()
                if task is None:
                    break
                image, save_path = task
                ext = os.path.splitext(save_path)[1]
                if not ext:
                    ext = '.jpg'
                ret, buffer = cv2.imencode(ext, image)
                if ret:
                    with open(save_path, 'wb') as f:
                        f.write(buffer)
                else:
                    print(f"[Worker Error] 图像编码失败：{save_path}")
            except Exception as e:
                print(f"[Worker Error] 保存截图失败：{e}")
            finally:
                try:
                    self.save_queue.task_done()
                except ValueError:
                    pass

    def launch_result_viewer(self, task_file_path: str):
        """启动结果查看器"""
        try:
            base_dir = get_base_path()
            viewer_exe = base_dir / VIEWER_EXE_NAME
            if not viewer_exe.exists():
                self.log_message(f"错误：viewer.exe 不存在于 {viewer_exe}")
                QMessageBox.critical(self, "错误", f"未找到 viewer.exe，请确保它与主程序在同一目录！")
                return
            subprocess.Popen([str(viewer_exe), "--task", task_file_path])
            self.log_message(f"已启动 viewer.exe，加载任务文件：{task_file_path}")
        except Exception as e:
            self.log_message(f"启动 viewer.exe 失败：{e}")
            QMessageBox.critical(self, "错误", f"无法启动结果查看器：\n{str(e)}")

    def load_configuration(self):
        """加载配置文件"""
        config = configparser.ConfigParser()
        config_path = get_base_path() / CONFIG_FILE_NAME
        if not config_path.exists():
            config['Settings'] = {
                'model_path': DEFAULT_MODEL_PATH,
                'nms_iou_threshold': str(DEFAULT_NMS_IOU_THRESHOLD),
                'inference_size': str(DEFAULT_INFERENCE_SIZE),
                'batch_size': str(DEFAULT_BATCH_SIZE),
                'roi_iou_threshold': str(DEFAULT_ROI_IOU_THRESHOLD),
                'movement_iou_threshold': str(DEFAULT_MOVEMENT_IOU_THRESHOLD),
                'movement_relative_threshold': str(DEFAULT_MOVEMENT_RELATIVE_THRESHOLD),
                'movement_consecutive_frames': str(DEFAULT_MOVEMENT_CONSECUTIVE_FRAMES),
                'movement_stable_seconds': str(DEFAULT_MOVEMENT_STABLE_SECONDS)
            }
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    config.write(f)
                self.log_message(f"已创建默认配置文件 {config_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法创建配置文件：{e}")
                return False
        else:
            config.read(config_path, encoding='utf-8')

        self.MODEL_PATH = config.get('Settings', 'model_path', fallback=DEFAULT_MODEL_PATH)
        try:
            self.NMS_IOU_THRESHOLD = float(config.get('Settings', 'nms_iou_threshold', fallback=str(DEFAULT_NMS_IOU_THRESHOLD)))
            self.NMS_IOU_THRESHOLD = max(0.0, min(1.0, self.NMS_IOU_THRESHOLD))
        except ValueError:
            self.NMS_IOU_THRESHOLD = DEFAULT_NMS_IOU_THRESHOLD
        try:
            self.INFERENCE_SIZE = int(config.get('Settings', 'inference_size', fallback=str(DEFAULT_INFERENCE_SIZE)))
            self.INFERENCE_SIZE = max(320, min(1280, self.INFERENCE_SIZE))
        except ValueError:
            self.INFERENCE_SIZE = DEFAULT_INFERENCE_SIZE
        try:
            self.BATCH_SIZE = int(config.get('Settings', 'batch_size', fallback=str(DEFAULT_BATCH_SIZE)))
            self.BATCH_SIZE = max(1, self.BATCH_SIZE)
        except ValueError:
            self.BATCH_SIZE = DEFAULT_BATCH_SIZE
        try:
            self.ROI_IOU_THRESHOLD = float(config.get('Settings', 'roi_iou_threshold', fallback=str(DEFAULT_ROI_IOU_THRESHOLD)))
        except ValueError:
            self.ROI_IOU_THRESHOLD = DEFAULT_ROI_IOU_THRESHOLD
        try:
            self.MOVEMENT_IOU_THRESHOLD = float(config.get('Settings', 'movement_iou_threshold', fallback=str(DEFAULT_MOVEMENT_IOU_THRESHOLD)))
        except ValueError:
            self.MOVEMENT_IOU_THRESHOLD = DEFAULT_MOVEMENT_IOU_THRESHOLD
        try:
            self.MOVEMENT_RELATIVE_THRESHOLD = float(config.get('Settings', 'movement_relative_threshold', fallback=str(DEFAULT_MOVEMENT_RELATIVE_THRESHOLD)))
        except ValueError:
            self.MOVEMENT_RELATIVE_THRESHOLD = DEFAULT_MOVEMENT_RELATIVE_THRESHOLD
        try:
            self.MOVEMENT_CONSECUTIVE_FRAMES = int(config.get('Settings', 'movement_consecutive_frames', fallback=str(DEFAULT_MOVEMENT_CONSECUTIVE_FRAMES)))
        except ValueError:
            self.MOVEMENT_CONSECUTIVE_FRAMES = DEFAULT_MOVEMENT_CONSECUTIVE_FRAMES
        try:
            self.MOVEMENT_STABLE_SECONDS = float(config.get('Settings', 'movement_stable_seconds', fallback=str(DEFAULT_MOVEMENT_STABLE_SECONDS)))
        except ValueError:
            self.MOVEMENT_STABLE_SECONDS = DEFAULT_MOVEMENT_STABLE_SECONDS
        
        self.log_message(f"配置文件读取成功")
        return True

    def load_model(self) -> bool:
        """加载 YOLO 模型"""
        global ULTRALYTICS_AVAILABLE
        if ULTRALYTICS_AVAILABLE is None:
            try:
                import torch
                from ultralytics import YOLO
                ULTRALYTICS_AVAILABLE = True
                self.torch = torch
            except ImportError as e:
                ULTRALYTICS_AVAILABLE = False
                self.log_message(f"导入失败：{e}")
                QMessageBox.critical(self, "错误", "缺少 PyTorch 或 Ultralytics！")
                return False
        
        if not ULTRALYTICS_AVAILABLE:
            return False

        self.force_cpu_mode = False
        if self.torch.cuda.is_available():
            try:
                major, minor = self.torch.cuda.get_device_capability(0)
                gpu_name = self.torch.cuda.get_device_name(0)
                total_vram_gb = self.torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                self.log_message(f"GPU 型号：{gpu_name} | GPU 显存：{total_vram_gb:.1f} GB | Compute Capability: {major}.{minor}")
                if major < 5:
                    self.log_message(f"GPU 架构太低，不支持程序编译 CUDA 版本（{self.torch.version.cuda}）")
                    self.force_cpu_mode = True
                else:
                    try:
                        _ = self.torch.tensor([1.0], device='cuda')
                        self.log_message(f"GPU 初始化成功 | 程序编译 CUDA 版本：{self.torch.version.cuda}")
                        if PYNVML_AVAILABLE and not self._nvml_initialized:
                            try:
                                pynvml.nvmlInit()
                                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                                self._nvml_initialized = True
                                self.log_message("GPU 显存监控已启用")
                            except Exception as e:
                                self.log_message(f"GPU 显存监控初始化失败：{e}")
                    except Exception as e:
                        self.log_message(f"CUDA 初始化失败：{e}，回退 CPU")
                        self.force_cpu_mode = True
            except Exception as e:
                self.log_message(f"GPU 检测异常：{e}，回退 CPU")
                self.force_cpu_mode = True
        else:
            self.log_message(f"未检测到支持 CUDA{self.torch.version.cuda} 显卡，使用 CPU 模式")
            self.force_cpu_mode = True

        try:
            if not os.path.exists(self.MODEL_PATH):
                raise FileNotFoundError(f"模型文件不存在：{self.MODEL_PATH}")
            
            from ultralytics import YOLO
            device = 'cuda' if (self.torch.cuda.is_available() and not self.force_cpu_mode) else 'cpu'
            self.model = YOLO(self.MODEL_PATH)
            
            self.log_message(f"模型【{self.MODEL_PATH}】加载成功 | 运行设备：{device.upper()}")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "illegal memory access" in error_msg or "invalid device function" in error_msg:
                self.log_message(f"GPU 架构不兼容（禁用 CUDA）: {e}")
                try:
                    self.force_cpu_mode = True
                    from ultralytics import YOLO
                    self.model = YOLO(self.MODEL_PATH)
                    self.log_message("紧急回退到 CPU 模式")
                    return True
                except Exception as e2:
                    self.log_message(f"CPU 模式加载失败：{e2}")
                    QMessageBox.critical(self, "错误", f"模型加载失败：\n{e2}")
                    return False
            else:
                self.log_message(f"模型加载失败：{e}")
                QMessageBox.critical(self, "错误", f"模型加载失败：\n{e}")
                return False

    def _auto_adjust_batch_size(self):
        if not hasattr(self, 'torch') or not self.torch.cuda.is_available() or self.force_cpu_mode:
            self.log_message("使用 CPU 推理，batch_size 强制设为 1")
            self.BATCH_SIZE = 1
            return
        self.log_message(f"使用配置 batch_size，设置为 {self.BATCH_SIZE}")

    def select_video(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择视频文件", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.mpeg *.mpg *.m1v *.m2v *.vob *.ts *.m2ts *.mts)")
        if paths:
            self.file_paths = list(paths)
            display = ", ".join(paths[:2]) + f", ... (+{len(paths) - 2})" if len(paths) > 3 else ", ".join(paths)
            self.video_entry.setText(display)
            self.log_message(f"已选择 {len(self.file_paths)} 个视频文件")

    def select_save_path(self):
        folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if folder:
            self.save_folder = folder
            self.save_path_entry.setText(folder)
            try:
                test = os.path.join(folder, "test.tmp")
                open(test, 'w').close()
                os.remove(test)
                self.log_message("路径验证：具有写入权限，支持中文路径")
            except Exception as e:
                QMessageBox.warning(self, "权限警告", f"保存路径可能有问题：{e}")

    def select_all_classes(self):
        for cb in self.class_checkboxes:
            cb.setChecked(True)
        self.log_message("全选所有类别")

    def deselect_all_classes(self):
        for cb in self.class_checkboxes:
            cb.setChecked(False)
        self.log_message("取消已选目标")

    def toggle_preview(self):
        if not self.file_paths:
            self.log_message("请先选择视频文件")
            return
        if self._preview_active:
            self._preview_active = False
            self.preview_button.setText("开启实时预览")
            self.preview_status.setText("预览：关闭")
            self.preview_status.setStyleSheet("color: gray;")
            self.preview_label.setText("预览已关闭")
            self.preview_timer.stop()
            self.log_message("实时预览已关闭")
        else:
            self._preview_active = True
            self.preview_button.setText("关闭实时预览")
            self.preview_status.setText("预览：开启")
            self.preview_status.setStyleSheet("color: orange;")
            self.log_message("实时预览已开启")
            self.preview_timer.start(30)

    def _update_preview_display(self):
        if not self._preview_active:
            return
        try:
            item = self._preview_queue.get_nowait()
            if isinstance(item, tuple) and len(item) == 3:
                frame, frame_idx, annotate_info = item
                if frame is not None:
                    if self.roi is not None:
                        cv2.polylines(frame, [self.roi], isClosed=True, color=(0, 255, 0), thickness=2)
                    for (x1, y1, x2, y2, class_display, score) in annotate_info:
                        color = COLORS[hash(class_display) % len(COLORS)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        class_en = self.CHINESE_TO_ENGLISH.get(class_display, class_display)
                        text = f"{class_en} {score:.2f}"
                        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        text_x = x1
                        text_y = y1 - 10 if y1 - 10 > 0 else y1 + text_h + 10
                        cv2.rectangle(frame, (text_x, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), color, -1)
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    qimg = cv2_to_qimage(frame)
                    pixmap = QPixmap.fromImage(qimg)
                    if not pixmap.isNull():
                        self.preview_label.setPixmap(pixmap.scaled(
                            self.preview_label.size(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        ))
                    else:
                        self.preview_label.setText("预览图像无效")
                else:
                    self.preview_label.setText("视频结束")
        except queue.Empty:
            pass
        except Exception as e:
            self.log_message(f"[预览错误] {e}")

    def select_roi(self):
        if not self.file_paths:
            self.log_message("错误：请先选择视频文件")
            return
        cap = cv2.VideoCapture(self.file_paths[0])
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.log_message("错误：无法读取视频首帧")
            return
        dialog = ROIDialog(self, frame)
        if dialog.exec():
            roi = dialog.get_roi()
            if roi is not None and len(roi) >= 3:
                area = cv2.contourArea(roi)
                if area > 1.0:
                    self.roi = roi
                    self.roi_label.setText("关注区域已设置")
                    self.roi_label.setStyleSheet("color: orange;")
                    self.roi_button.setText("取消选取")
                    self.roi_button.clicked.disconnect()
                    self.roi_button.clicked.connect(self.cancel_roi)
                    self.log_message(f"成功设置 ROI，面积：{area:.1f} 像素")
                else:
                    QMessageBox.warning(self, "ROI 无效", "ROI 面积过小")
            else:
                QMessageBox.warning(self, "ROI 无效", "至少需要 3 个点")
        else:
            self.log_message("ROI 选择已取消")

    def cancel_roi(self):
        self.roi = None
        self.roi_label.setText("未选取关注区域")
        self.roi_label.setStyleSheet("color: gray;")
        self.roi_button.setText("选取关注区域")
        self.roi_button.clicked.disconnect()
        self.roi_button.clicked.connect(self.select_roi)
        self.log_message("已取消关注区域")

    def _box_iou_simple(self, box1, box2) -> float:
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
        return inter_area / union_area if union_area > 0 else 0.0

    def _is_point_in_roi(self, x, y) -> bool:
        if self.roi is None:
            return True
        try:
            pt = (int(round(x)), int(round(y)))
            result = cv2.pointPolygonTest(self.roi, pt, measureDist=False)
            return result >= 0
        except Exception as e:
            self.log_message(f"[ROI 安全防护] 检查点 ({x:.1f},{y:.1f}) 时出错：{e}")
            return True

    def _match_tracks(self, current_boxes, prev_tracks, iou_threshold=0.3):
        assigned_tracks = set()
        results = []
        
        for box in current_boxes:
            best_iou = 0
            best_tid = None
            
            for tid, data in prev_tracks.items():
                if tid in assigned_tracks:
                    continue
                iou = self._box_iou_simple(box, data['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            
            if best_iou >= iou_threshold and best_tid is not None:
                assigned_tracks.add(best_tid)
                results.append((box, best_tid, False))
            else:
                new_tid = f"track_{time.time()}_{len(results)}"
                results.append((box, new_tid, True))
        
        return results

    def _is_target_moving_enhanced(self, track_id, current_box, current_time, prev_tracks) -> bool:
        prev_data = prev_tracks.get(track_id)
        if prev_data is None:
            prev_tracks[track_id] = {'box': current_box, 'time': current_time}
            return True
        
        prev_box = prev_data['box']
        prev_time = prev_data['time']
        time_diff = current_time - prev_time
        
        if time_diff <= 0:
            return False
        
        if time_diff >= 0.3:
            cx1, cy1 = (prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2
            cx2, cy2 = (current_box[0] + current_box[2]) / 2, (current_box[1] + current_box[3]) / 2
            dx, dy = abs(cx2 - cx1), abs(cy2 - cy1)
            w = (current_box[2] - current_box[0] + prev_box[2] - prev_box[0]) / 2 + 1e-6
            h = (current_box[3] - current_box[1] + prev_box[3] - prev_box[1]) / 2 + 1e-6
            speed_x = dx / time_diff / w
            speed_y = dy / time_diff / h
            
            if speed_x > 1.0 or speed_y > 1.0:
                prev_tracks[track_id] = {'box': current_box, 'time': current_time}
                return True
            
            iou = self._box_iou_simple(prev_box, current_box)
            if iou > self.MOVEMENT_IOU_THRESHOLD:
                prev_tracks[track_id] = {'box': current_box, 'time': current_time}
                return False
            
            w_avg = ((current_box[2] - current_box[0]) + (prev_box[2] - prev_box[0])) / 2 + 1e-6
            h_avg = ((current_box[3] - current_box[1]) + (prev_box[3] - prev_box[1])) / 2 + 1e-6
            dx = abs((current_box[0] + current_box[2]) / 2 - (prev_box[0] + prev_box[2]) / 2)
            dy = abs((current_box[1] + current_box[3]) / 2 - (prev_box[1] + prev_box[3]) / 2)
            rel_dx, rel_dy = dx / w_avg, dy / h_avg
            
            moved = rel_dx > self.MOVEMENT_RELATIVE_THRESHOLD or rel_dy > self.MOVEMENT_RELATIVE_THRESHOLD
            prev_tracks[track_id] = {'box': current_box, 'time': current_time}
            return moved
        
        prev_tracks[track_id] = {'box': current_box, 'time': current_time}
        return False

    def start_processing(self):
        if not self.file_paths:
            self.log_message("错误：未选择视频文件")
            return
        if not self.save_folder:
            self.log_message("错误：未选择保存路径")
            return
        try:
            test = os.path.join(self.save_folder, "test.tmp")
            open(test, 'w').close()
            os.remove(test)
            os.makedirs(self.save_folder, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "路径错误", f"保存路径不可写：{e}")
            return
        
        selected = [self.class_names[i] for i, cb in enumerate(self.class_checkboxes) if cb.isChecked()]
        if not selected:
            self.log_message("错误：未选择任何目标类")
            return
        
        try:
            skip_frames = max(1, min(64, int(self.speed_entry.text())))
        except ValueError:
            skip_frames = 1
        try:
            conf = max(0.0, min(1.0, float(self.confidence_entry.text())))
        except ValueError:
            conf = 0.1
        
        only_moving = self.only_moving_var.isChecked()
        annotate = self.annotate_var.isChecked()
        
        self._auto_adjust_batch_size()
        self.paused = False
        self.stopped = False
        self._processing_finished_normally = False
        self._generated_screenshot_paths.clear()
        
        self.button_state_signal.emit(False, True, True)
        self.log_message("开始处理...")
        threading.Thread(target=self.process_videos, args=(
            self.file_paths, self.save_folder, selected, skip_frames, only_moving, annotate, conf
        ), daemon=True).start()

    def pause_processing(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.setText("继续处理")
            self.log_message("处理已暂停")
            self.update_status("处理已暂停...")
        else:
            self.pause_button.setText("暂停处理")
            self.log_message("处理已继续")
            self.update_status("处理中...")

    def stop_processing(self):
        if not self.stopped:
            self.stopped = True
            self.paused = False
            self.log_message("正在停止处理...")
            self.update_status("正在停止...")
            while not self.save_queue.empty():
                try:
                    self.save_queue.get_nowait()
                    self.save_queue.task_done()
                except:
                    break
            self.button_state_signal.emit(True, False, False)

    def process_videos(self, file_paths, save_folder, selected_classes, skip_frames, only_moving, annotate_objects, confidence_threshold):
        try:
            total_files = len(file_paths)
            self._total_files = total_files
            for file_idx, file_path in enumerate(file_paths):
                if self.stopped:
                    break
                self._processing_file_index = file_idx
                self.current_frame = 0
                cap_temp = cv2.VideoCapture(file_path)
                total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
                orig_h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                orig_w = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap_temp.release()
                self.total_frames = total_frames
                self._progress_total = total_frames
                
                prev_tracks = {}
                
                frame_queue = queue.Queue(maxsize=10)
                
                def read_frames():
                    cap = cv2.VideoCapture(file_path)
                    frame_idx = 0
                    while cap.isOpened() and not self.stopped:
                        while self.paused:
                            time.sleep(0.01)
                        if self.stopped:
                            break
                        ret = cap.grab()
                        if not ret:
                            break
                        if frame_idx % skip_frames == 0:
                            ret, frame = cap.retrieve()
                            if ret:
                                self.current_frame = frame_idx
                                try:
                                    frame_queue.put((frame.copy(), frame_idx), timeout=1)
                                except queue.Full:
                                    time.sleep(0.01)
                        frame_idx += 1
                    cap.release()
                    frame_queue.put(None)
                
                reader_thread = threading.Thread(target=read_frames, daemon=True)
                reader_thread.start()
                
                batch_original_frames = []
                batch_frame_counts = []
                last_inference_time = time.time()
                
                while True:
                    if self.stopped:
                        break
                    try:
                        item = frame_queue.get(timeout=2)
                        if item is None:
                            break
                    except queue.Empty:
                        continue
                    
                    orig_frame, frame_count = item
                    batch_original_frames.append(orig_frame)
                    batch_frame_counts.append(frame_count)
                    self._progress_current = frame_count
                    
                    if len(batch_original_frames) >= self.BATCH_SIZE:
                        self._run_batch_inference_optimized(
                            batch_original_frames, batch_frame_counts,
                            selected_classes, only_moving, annotate_objects,
                            confidence_threshold, orig_w, orig_h,
                            save_folder, file_path,
                            prev_tracks
                        )
                        current_time = time.time()
                        if current_time > last_inference_time:
                            self.stats['fps'] = len(batch_frame_counts) / (current_time - last_inference_time)
                            last_inference_time = current_time
                        batch_original_frames.clear()
                        batch_frame_counts.clear()
                
                if batch_original_frames:
                    self._run_batch_inference_optimized(
                        batch_original_frames, batch_frame_counts,
                        selected_classes, only_moving, annotate_objects,
                        confidence_threshold, orig_w, orig_h,
                        save_folder, file_path,
                        prev_tracks
                    )
                
                reader_thread.join(timeout=2)
                
        except Exception as e:
            self.log_message(f"处理出错：{e}\n{traceback.format_exc()}")
            self.update_status(f"错误：{str(e)}")
        finally:
            self.button_state_signal.emit(True, False, False)
            
            if self.stopped:
                self.update_status("处理已停止")
                self.log_message("处理已停止")
            else:
                self._processing_finished_normally = True
                self.update_status("所有文件处理完成！")
                task_file = self._generate_task_summary_file()
                self.processing_finished_signal.emit(task_file if task_file else "")
            
            if hasattr(self, 'torch') and self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

    def _run_batch_inference_optimized(self, batch_original_frames, batch_frame_counts, selected_classes, only_moving, annotate_objects, confidence_threshold, orig_w, orig_h, save_folder, file_path, prev_tracks):
        try:
            device = 'cuda' if (hasattr(self, 'torch') and self.torch.cuda.is_available() and not self.force_cpu_mode) else 'cpu'
            results_list = self.model(
                batch_original_frames,
                conf=confidence_threshold,
                iou=self.NMS_IOU_THRESHOLD,
                verbose=False,
                device=device
            )
            
            for idx, results in enumerate(results_list):
                frame = batch_original_frames[idx]
                frame_count = batch_frame_counts[idx]
                current_time = time.time()
                
                if not (results and len(results.boxes) > 0):
                    if self._preview_active:
                        try:
                            if self._preview_queue.full():
                                self._preview_queue.get_nowait()
                            self._preview_queue.put_nowait((frame.copy(), frame_count, []))
                        except queue.Full:
                            pass
                    continue
                
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                filtered = []
                for box, score, cls_id in zip(boxes, scores, class_ids):
                    if cls_id >= len(COCO_CLASSES):
                        continue
                    class_name = COCO_CLASSES[cls_id]
                    if class_name not in selected_classes:
                        continue
                    x1, y1, x2, y2 = box.astype(int)
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if self.roi is not None:
                        if not self._is_point_in_roi(cx, cy):
                            continue
                    filtered.append((box, score, cls_id))
                
                if not filtered:
                    if self._preview_active:
                        try:
                            if self._preview_queue.full():
                                self._preview_queue.get_nowait()
                            self._preview_queue.put_nowait((frame.copy(), frame_count, []))
                        except queue.Full:
                            pass
                    continue
                
                detections = sorted(filtered, key=lambda x: x[1], reverse=True)
                to_keep = [True] * len(detections)
                for i in range(len(detections)):
                    if not to_keep[i]: continue
                    box_i, _, cls_i = detections[i]
                    for j in range(i + 1, len(detections)):
                        if not to_keep[j]: continue
                        box_j, _, cls_j = detections[j]
                        if cls_i == cls_j:
                            iou = self._box_iou_simple(box_i, box_j)
                            if iou > 0.85:
                                to_keep[j] = False
                kept = [detections[i] for i in range(len(detections)) if to_keep[i]]
                
                current_boxes = [d[0] for d in kept]
                matched_results = self._match_tracks(current_boxes, prev_tracks, iou_threshold=0.3)
                
                preview_annotate_info = []
                targets_to_save = []
                
                for i, (box, track_id, is_new) in enumerate(matched_results):
                    _, score, cls_id = kept[i]
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = COCO_CLASSES[cls_id]
                    class_display = COCO_CLASS_MAPPING.get(class_name, class_name)
                    
                    should_save = True
                    if only_moving:
                        should_save = self._is_target_moving_enhanced(track_id, [x1, y1, x2, y2], current_time, prev_tracks)
                    
                    if should_save:
                        targets_to_save.append((x1, y1, x2, y2, class_display, score))
                        preview_annotate_info.append((x1, y1, x2, y2, class_display, float(score)))
                
                if targets_to_save:
                    video_name = Path(file_path).stem
                    for i, (x1, y1, x2, y2, class_display, score) in enumerate(targets_to_save):
                        if annotate_objects:
                            annotated_frame = frame.copy()
                            color = COLORS[hash(class_display) % len(COLORS)]
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            class_en = self.CHINESE_TO_ENGLISH.get(class_display, class_display)
                            text = f"{class_en} {score:.2f}"
                            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            text_x = x1
                            text_y = y1 - 10 if y1 - 10 > 0 else y1 + text_h + 10
                            cv2.rectangle(annotated_frame, (text_x, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), color, -1)
                            cv2.putText(annotated_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            image_to_save = annotated_frame
                        else:
                            image_to_save = frame.copy()
                        
                        safe_class = sanitize_filename(class_display)
                        filename = f"{video_name}_frame{frame_count}_{safe_class}.jpg"
                        save_path = os.path.join(save_folder, filename)
                        try:
                            self.save_queue.put_nowait((image_to_save, save_path))
                            self.stats['total_targets'] += 1
                            self.stats['targets_by_class'][class_display] = \
                                self.stats['targets_by_class'].get(class_display, 0) + 1
                            self._generated_screenshot_paths.append(os.path.abspath(save_path))
                        except queue.Full:
                            pass
                
                if self._preview_active:
                    try:
                        if self._preview_queue.full():
                            self._preview_queue.get_nowait()
                        self._preview_queue.put_nowait((frame.copy(), frame_count, preview_annotate_info))
                    except queue.Full:
                        pass
        except Exception as e:
            self.log_message(f"批量推理出错：{e}\n{traceback.format_exc()}")

    def _update_progress_bar(self):
        if self._processing_finished_normally:
            self.progress_bar.setValue(100)
            self.status_label.setText("所有文件处理完成！")
            return
        if self.stopped:
            status_text = "处理已停止"
        elif self._total_files <= 0:
            status_text = "准备中..."
        else:
            file_progress = self._progress_current / self._progress_total if self._progress_total > 0 else 0
            overall_progress = (self._processing_file_index + file_progress) / self._total_files
            progress_percent = min(99, int(overall_progress * 100 + 0.5))
            self.progress_bar.setValue(progress_percent)
            status_text = f"处理文件 {self._processing_file_index + 1}/{self._total_files} | 帧：{self._progress_current}/{self._progress_total}"
        if not self._processing_finished_normally:
            self.status_label.setText(status_text)

    def update_stats_periodically(self):
        self.fps_label.setText(f"FPS(每秒处理帧数): {self.stats['fps']:.1f}")
        self.targets_label.setText(f"截图数（未去重）: {self.stats['total_targets']}")
        if self.gpu_handle is None and PYNVML_AVAILABLE and not self._nvml_initialized:
            if self.torch is not None and hasattr(self.torch, 'cuda'):
                if self.torch.cuda.is_available() and not getattr(self, 'force_cpu_mode', False):
                    try:
                        pynvml.nvmlInit()
                        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        self._nvml_initialized = True  # 标记已初始化
                        self.log_message("GPU 显存监控已启用")
                    except Exception as e:
                        self.gpu_handle = None
                        if not hasattr(self, '_nvml_error_logged'):
                            self._nvml_error_logged = True
                            self.log_message(f"GPU 显存监控初始化失败：{e}")
        
        if self.gpu_handle is not None:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem_gb = mem_info.used / (1024 ** 3)
                self.gpu_label.setText(f"已用 GPU 显存：{gpu_mem_gb:.1f} GB")
            except Exception as e:
                self.gpu_label.setText("已用 GPU 显存：错误")
        else:
            self.gpu_label.setText("已用 GPU 显存：N/A")

    def _generate_task_summary_file(self) -> str:
        if not self._generated_screenshot_paths:
            self.log_message("无截图生成，跳过任务记录文件创建。")
            return ""
        unique_paths = list(dict.fromkeys(self._generated_screenshot_paths))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task_filename = f"任务_{timestamp}.txt"
        task_filepath = os.path.join(self.save_folder, task_filename)
        try:
            with open(task_filepath, "w", encoding="utf-8") as f:
                f.write("[Processed Videos]\n")
                for video_path in self.file_paths:
                    f.write(f"{os.path.abspath(video_path)}\n")
                f.write("\n[Generated Screenshots]\n")
                for img_path in unique_paths:
                    f.write(f"{img_path}\n")
            self.log_message(f"任务记录文件已生成（已去重）: {task_filepath}")
            return os.path.abspath(task_filepath)
        except Exception as e:
            self.log_message(f"生成任务记录文件失败：{e}")
            QMessageBox.warning(self, "警告", f"无法创建任务记录文件：\n{e}")
            return ""

    def closeEvent(self, event):
        self.stopped = True
        if hasattr(self, 'stats_timer'):
            self.stats_timer.stop()
        if hasattr(self, 'preview_timer'):
            self.preview_timer.stop()
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()
        self.save_queue.put(None)
        if PYNVML_AVAILABLE and self.gpu_handle is not None:
            try:
                pynvml.nvmlShutdown()
                self.log_message("GPU 监控已关闭")
            except:
                pass
        event.accept()

    def on_processing_finished(self, task_file_path: str):
        """主线程中安全地创建 [查看结果] 按钮"""
        if hasattr(self, '_view_results_button') and self._view_results_button:
            self._view_results_button.deleteLater()
            self._view_results_button = None
        if not task_file_path:
            return
        self._view_results_button = QPushButton("查看结果")
        self._view_results_button.clicked.connect(
            lambda: self.launch_result_viewer(task_file_path)
        )
        if hasattr(self, 'right_layout') and self.right_layout:
            self.right_layout.insertWidget(2, self._view_results_button)
        else:
            self.log_message("警告：无法定位右侧布局")

def main():
    app = QApplication(sys.argv)
    splash_pix = QPixmap(400, 200)
    splash_pix.fill(Qt.white)
    painter = QPainter(splash_pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor(66, 133, 244), 3))
    painter.drawRect(10, 10, 380, 180)
    painter.setPen(QColor(33, 33, 33))
    painter.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
    painter.drawText(QRect(0, 40, 400, 40), Qt.AlignCenter, "视频目标检测截图工具")
    painter.setFont(QFont("Microsoft YaHei", 10))
    painter.drawText(QRect(0, 90, 400, 30), Qt.AlignCenter, "V3.0 by geckotao")
    painter.setPen(QColor(220, 50, 47))
    painter.drawText(QRect(0, 130, 400, 30), Qt.AlignCenter, "⏳ 程序初始化中...")
    painter.end()
    splash = QSplashScreen(splash_pix)
    splash.show()
    app.processEvents()
    window = MainWindow()
    window.showMaximized()
    splash.finish(window)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
