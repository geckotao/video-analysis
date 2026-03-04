# View_results.py
import sys
import os
import re
import cv2
import time
import numpy as np
from collections import OrderedDict
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QListWidget, QFileDialog,
    QMessageBox, QSpinBox, QSizePolicy, QScrollArea,
    QCheckBox, QListView, QStyledItemDelegate, QStyle
)
from PySide6.QtCore import (
    QTimer, Qt, QRect, QSize, QAbstractListModel, QModelIndex
)
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFontMetrics, QIcon, QFont
)

# ==================== 常量定义 ====================
DEFAULT_FPS = 30.0
DEFAULT_PLAYBACK_SPEEDS = [
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0,
    2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0
]
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".mpeg", ".mpg", ".m1v", ".m2v", ".vob", ".ts", ".m2ts", ".mts")
CACHE_MAX_SIZE = 100  # 图片缓存最大数量

# 颜色常量
COLOR_BG_DEFAULT = QColor(45, 45, 45)
COLOR_TEXT_DEFAULT = QColor(180, 180, 180)
COLOR_BG_ITEM = QColor(40, 40, 40)
COLOR_BG_ITEM_SELECTED = QColor(60, 60, 60)
COLOR_TEXT_ITEM = QColor(220, 220, 220)
COLOR_TEXT_ERROR = QColor(220, 50, 50)
COLOR_TIMELINE_BG = QColor(80, 80, 80)
COLOR_TIMELINE_PLAYED = QColor(70, 130, 180)
COLOR_TIMELINE_THUMB = QColor(220, 220, 220)
COLOR_TIMELINE_TICK = QColor(180, 180, 180)

# 布局常量
LIST_ITEM_MIN_WIDTH = 320
LIST_ITEM_IMG_RATIO = 0.75
LIST_ITEM_TEXT_HEIGHT = 18
LIST_ITEM_PADDING = 6
TIMELINE_HEIGHT_MIN = 40
TIMELINE_HEIGHT_MAX = 100
TIMELINE_MARGIN = 20
TIMELINE_TRACK_HEIGHT = 4
TIMELINE_THUMB_RADIUS = 8

# ==================== 自适应图片列表视图 ====================
class AdaptiveListView(QListView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setResizeMode(QListView.Adjust)
        self.setWrapping(True)
        self.setWordWrap(False)
        self.setUniformItemSizes(True)
        self.setSpacing(10)
        self.setSelectionMode(QListView.NoSelection)
        self.setViewMode(QListView.IconMode)
        self.setStyleSheet("QListView { background: #282828; border: none; }")
        self.setContentsMargins(0, 0, 0, 0)
        self._delegate = None

    def setItemDelegate(self, delegate):
        super().setItemDelegate(delegate)
        self._delegate = delegate

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_grid_size()

    def _update_grid_size(self):
        viewport_width = self.viewport().width()
        spacing = self.spacing()
        if viewport_width <= 0:
            return
        num_columns = max(1, (viewport_width + spacing) // (LIST_ITEM_MIN_WIDTH + spacing))
        item_width = (viewport_width - (num_columns - 1) * spacing) // num_columns
        img_height = int(item_width * LIST_ITEM_IMG_RATIO)
        item_height = img_height + LIST_ITEM_TEXT_HEIGHT + LIST_ITEM_PADDING
        self.setGridSize(QSize(item_width, item_height))
        if self._delegate and hasattr(self._delegate, '_set_item_size'):
            self._delegate._set_item_size(QSize(item_width, item_height))

# ==================== 图片模型 ====================
class ImageListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []  # (img_path, video_name, frame_num, target_name)
        self._visible_indices = []
        self._allowed_targets = set()

    def set_items(self, items):
        self.beginResetModel()
        self._items = items
        self._allowed_targets = set()
        self._update_visible()
        self.endResetModel()

    def set_allowed_targets(self, allowed_targets):
        self.beginResetModel()
        self._allowed_targets = set(allowed_targets) if allowed_targets else set()
        self._update_visible()
        self.endResetModel()

    def _update_visible(self):
        if not self._allowed_targets:
            self._visible_indices = list(range(len(self._items)))
        else:
            self._visible_indices = [
                i for i, item in enumerate(self._items)
                if item[3] in self._allowed_targets
            ]

    def get_item(self, index):
        if not index.isValid():
            return None
        real_row = self._visible_indices[index.row()]
        return self._items[real_row]

    def rowCount(self, parent=QModelIndex()):
        return len(self._visible_indices)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.UserRole:
            return self.get_item(index)
        return None

# ==================== 图片委托 ====================
class ImageItemDelegate(QStyledItemDelegate):
    def __init__(self, video_player, parent=None):
        super().__init__(parent)
        self.video_player = video_player
        self._pixmap_cache = OrderedDict()
        self._current_item_size = QSize(200, 260)
        # 缓存绘图对象
        self._pen_text = QPen(COLOR_TEXT_ITEM)
        self._pen_error = QPen(COLOR_TEXT_ERROR)
        self._font_text = QFont()
        self._font_text.setPointSize(14)

    def _set_item_size(self, size):
        self._current_item_size = size

    def paint(self, painter, option, index):
        item = index.data(Qt.UserRole)
        if not item:
            return
        img_path, video_name, frame_num, target_name = item
        
        bg_color = COLOR_BG_ITEM_SELECTED if option.state & QStyle.State_Selected else COLOR_BG_ITEM
        painter.fillRect(option.rect, bg_color)
        
        max_img_width = self._current_item_size.width() - 10
        pixmap = self._load_pixmap(img_path, max_img_width)
        
        painter.setFont(self._font_text)
        if pixmap:
            x = option.rect.left() + (option.rect.width() - pixmap.width()) // 2
            img_y_offset = 4
            y = option.rect.top() + img_y_offset
            painter.drawPixmap(x, y, pixmap)
            
            text_height = LIST_ITEM_TEXT_HEIGHT
            text_y = y + pixmap.height() + 2
            text_rect = QRect(option.rect.left(), text_y, option.rect.width(), text_height)
            painter.setPen(self._pen_text)
            painter.drawText(text_rect, Qt.AlignCenter, target_name)
        else:
            painter.setPen(self._pen_error)
            painter.drawText(option.rect, Qt.AlignCenter, "加载失败")

    def sizeHint(self, option, index):
        return self._current_item_size

    def _load_pixmap(self, img_path, max_width):
        cache_key = (img_path, max_width)
        if cache_key in self._pixmap_cache:
            # Move to end for LRU
            self._pixmap_cache.move_to_end(cache_key)
            return self._pixmap_cache[cache_key]
        
        try:
            with open(img_path, 'rb') as f:
                img_data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("decode failed")
            
            h, w = img.shape[:2]
            scale = max_width / w
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            
            # Cache management
            if len(self._pixmap_cache) >= CACHE_MAX_SIZE:
                self._pixmap_cache.popitem(last=False)
            self._pixmap_cache[cache_key] = pixmap
            return pixmap
        except Exception:
            if len(self._pixmap_cache) >= CACHE_MAX_SIZE:
                self._pixmap_cache.popitem(last=False)
            self._pixmap_cache[cache_key] = None
            return None

# ==================== 时间轴 ====================
class TimelineWidget(QWidget):
    def __init__(self, video_player, parent=None):
        super().__init__(parent)
        self.video_player = video_player
        self.setMinimumHeight(TIMELINE_HEIGHT_MIN)
        self.setMaximumHeight(TIMELINE_HEIGHT_MAX)
        self._total_frames = 0
        self._current_frame = 0
        self._fps = DEFAULT_FPS
        self._is_dragging = False
        self._track_rect = QRect()
        self._thumb_radius = TIMELINE_THUMB_RADIUS
        # 缓存绘图对象
        self._pen_track = QPen(Qt.NoPen)
        self._brush_track = QColor(80, 80, 80)
        self._brush_played = QColor(70, 130, 180)
        self._brush_thumb = QColor(220, 220, 220)
        self._pen_tick = QPen(COLOR_TIMELINE_TICK)
        self._font_time = QFont()
        self._font_time.setPointSize(8)

    def set_total_frames(self, total):
        self._total_frames = max(1, total)
        self.update()

    def set_current_frame(self, frame):
        self._current_frame = max(0, min(frame, self._total_frames - 1))
        self.update()

    def get_current_frame(self):
        return self._current_frame

    def set_fps(self, fps):
        self._fps = max(0.1, fps)

    def _frame_to_x(self, frame):
        if self._total_frames <= 1:
            return self._track_rect.left() + self._track_rect.width() // 2
        ratio = frame / (self._total_frames - 1)
        return self._track_rect.left() + int(ratio * self._track_rect.width())

    def _x_to_frame(self, x):
        if self._total_frames <= 1:
            return 0
        if x <= self._track_rect.left():
            return 0
        if x >= self._track_rect.right():
            return self._total_frames - 1
        ratio = (x - self._track_rect.left()) / self._track_rect.width()
        return int(round(ratio * (self._total_frames - 1)))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.video_player._save_play_state()
            self._is_dragging = True
            x = int(event.position().x())
            self._current_frame = self._x_to_frame(x)
            self.update()
            self.video_player.timeline_changed(self._current_frame)
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_dragging:
            x = int(event.position().x())
            self._current_frame = self._x_to_frame(x)
            self.update()
            self.video_player.timeline_changed(self._current_frame)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_dragging:
            self._is_dragging = False
            self.video_player.timeline_released(self._current_frame)
            event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        margin = TIMELINE_MARGIN
        track_height = TIMELINE_TRACK_HEIGHT
        self._track_rect = QRect(
            margin,
            self.height() // 2 - track_height // 2,
            self.width() - 2 * margin,
            track_height
        )
        painter.setPen(self._pen_track)
        painter.setBrush(self._brush_track)
        painter.drawRect(self._track_rect)
        
        if self._total_frames > 0:
            played_ratio = self._current_frame / self._total_frames
            played_width = int(self._track_rect.width() * played_ratio)
            played_rect = QRect(self._track_rect.left(), self._track_rect.top(), played_width, track_height)
            painter.setBrush(self._brush_played)
            painter.drawRect(played_rect)
            
            thumb_x = self._frame_to_x(self._current_frame)
            painter.setBrush(self._brush_thumb)
            painter.drawEllipse(
                thumb_x - self._thumb_radius,
                self.height() // 2 - self._thumb_radius,
                self._thumb_radius * 2,
                self._thumb_radius * 2
            )
            
            if self._total_frames <= 0 or self._fps <= 0:
                return
            total_seconds = self._total_frames / self._fps
            if total_seconds < 1:
                return
            
            num_ticks = 2
            if total_seconds <= 10: num_ticks = 2
            elif total_seconds <= 30: num_ticks = 3
            elif total_seconds <= 60: num_ticks = 4
            elif total_seconds <= 180: num_ticks = 5
            else: num_ticks = 6
            
            painter.setFont(self._font_time)
            metrics = QFontMetrics(self._font_time)
            painter.setPen(self._pen_tick)
            
            for i in range(num_ticks):
                t_sec = i * (total_seconds / (num_ticks - 1)) if num_ticks > 1 else 0
                frame_at_tick = int(t_sec * self._fps)
                x = self._frame_to_x(frame_at_tick)
                painter.drawLine(x, self._track_rect.bottom() + 2, x, self._track_rect.bottom() + 5)
                time_str = self._format_time(t_sec)
                text_width = metrics.horizontalAdvance(time_str)
                text_x = x - text_width // 2
                text_y = self._track_rect.bottom() + 15
                painter.drawText(text_x, text_y, time_str)

    def _format_time(self, seconds):
        if seconds < 0: seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

# ==================== 主窗口 ====================
class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("viewer.ico"))
        self.setWindowTitle("目标检测结果查看工具 V2.0")
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint |
            Qt.WindowSystemMenuHint
        )
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = DEFAULT_FPS
        self.is_playing = False
        self.play_direction = 1
        self.playback_speed = 1.0
        self.video_files = []
        self.current_video_index = -1
        self.was_playing_before_slider = False
        self.target_checkboxes = {}
        self.filter_layout = None
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_hbox = QHBoxLayout(central)
        main_hbox.setSpacing(10)
        main_hbox.setContentsMargins(10, 10, 10, 10)
        
        # 左侧图片区
        img_widget = QWidget()
        img_layout = QVBoxLayout(img_widget)
        btn_load_imgs = QPushButton("选择图片文件夹")
        btn_load_imgs.clicked.connect(self.load_images_from_folder)
        img_layout.addWidget(btn_load_imgs)
        btn_load_task = QPushButton("加载任务文件 (.txt)")
        btn_load_task.clicked.connect(self.load_task_file)
        img_layout.addWidget(btn_load_task)
        
        self.filter_layout = QHBoxLayout()
        self.filter_layout.setSpacing(10)
        self.filter_layout.setContentsMargins(5, 5, 5, 5)
        filter_widget = QWidget()
        filter_widget.setLayout(self.filter_layout)
        filter_scroll = QScrollArea()
        filter_scroll.setWidgetResizable(True)
        filter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        filter_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        filter_scroll.setMaximumHeight(40)
        filter_scroll.setWidget(filter_widget)
        img_layout.addWidget(filter_scroll)
        
        self.image_list_view = AdaptiveListView()
        self.image_list_view.clicked.connect(self.on_image_clicked)
        img_layout.addWidget(self.image_list_view, stretch=1)
        main_hbox.addWidget(img_widget, stretch=1)
        
        # 右侧复合区
        right_compound_widget = QWidget()
        right_compound_layout = QVBoxLayout(right_compound_widget)
        right_compound_layout.setSpacing(10)
        right_compound_layout.setContentsMargins(0, 0, 0, 0)
        
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        video_layout.setSpacing(8)
        video_layout.setContentsMargins(5, 5, 5, 5)
        self.video_label = VideoLabel()
        video_layout.addWidget(self.video_label, stretch=1)
        self.timeline = TimelineWidget(self)
        video_layout.addWidget(self.timeline)
        self.frame_info_label = QLabel("0 / 0    |    00:00 / 00:00")
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.frame_info_label)
        
        ctrl_layout = QHBoxLayout()
        self.btn_prev = QPushButton("上一帧")
        self.btn_reverse = QPushButton("倒放")
        self.btn_pause = QPushButton("暂停")
        self.btn_play = QPushButton("播放")
        self.btn_next = QPushButton("下一帧")
        self.btn_screenshot = QPushButton("截图")
        self.btn_play.clicked.connect(self.play_forward)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_reverse.clicked.connect(self.play_reverse)
        self.btn_prev.clicked.connect(self.step_backward)
        self.btn_next.clicked.connect(self.step_forward)
        self.btn_screenshot.clicked.connect(self.capture_screenshot)
        ctrl_layout.addWidget(self.btn_prev)
        ctrl_layout.addWidget(self.btn_reverse)
        ctrl_layout.addWidget(self.btn_pause)
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_next)
        ctrl_layout.addWidget(self.btn_screenshot)
        
        speed_layout = QHBoxLayout()
        self.btn_speed_down = QPushButton("减速")
        self.speed_label = QLabel("1x")
        self.speed_label.setAlignment(Qt.AlignCenter)
        self.speed_label.setFixedWidth(50)
        self.btn_speed_up = QPushButton("加速")
        self.btn_speed_down.setFixedWidth(30)
        self.btn_speed_up.setFixedWidth(30)
        self.btn_speed_down.clicked.connect(self.decrease_speed)
        self.btn_speed_up.clicked.connect(self.increase_speed)
        speed_layout.addWidget(self.btn_speed_down)
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.btn_speed_up)
        ctrl_layout.addLayout(speed_layout)
        video_layout.addLayout(ctrl_layout)
        
        jump_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("输入文件名（例如：demo.mp4）或双击视频列表选择文件名")
        self.frame_input = QSpinBox()
        self.frame_input.setRange(0, 999999)
        self.btn_jump = QPushButton("跳转到帧")
        self.btn_jump.clicked.connect(self.jump_to_frame)
        jump_layout.addWidget(QLabel("文件："))
        jump_layout.addWidget(self.file_input, stretch=1)
        jump_layout.addWidget(QLabel("帧："))
        jump_layout.addWidget(self.frame_input)
        jump_layout.addWidget(self.btn_jump)
        video_layout.addLayout(jump_layout)
        right_compound_layout.addWidget(video_widget, stretch=7.5)
        
        video_list_widget = QWidget()
        video_list_layout = QVBoxLayout(video_list_widget)
        video_list_layout.setSpacing(5)
        video_list_layout.setContentsMargins(5, 5, 5, 5)
        list_label = QLabel("视频列表")
        list_label.setAlignment(Qt.AlignCenter)
        video_list_layout.addWidget(list_label)
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.on_video_selected)
        video_list_layout.addWidget(self.list_widget, stretch=1)
        btn_add = QPushButton("添加视频")
        btn_add.clicked.connect(self.add_videos)
        video_list_layout.addWidget(btn_add)
        right_compound_layout.addWidget(video_list_widget, stretch=2.5)
        main_hbox.addWidget(right_compound_widget, stretch=1)
        
        self.set_playback_speed(1.0)
        self.image_model = ImageListModel()
        self.image_delegate = ImageItemDelegate(self)
        self.image_list_view.setModel(self.image_model)
        self.image_list_view.setItemDelegate(self.image_delegate)

    def match_video_by_basename(self, base_name):
        for video_path in self.video_files:
            video_basename = os.path.basename(video_path)
            video_base = os.path.splitext(video_basename)[0]
            if video_base == base_name:
                return video_basename
        print(f"警告：未找到与基础名 '{base_name}' 匹配的视频文件")
        return base_name

    def _parse_task_content(self, content):
        video_section = re.search(r'\[Processed Videos\]\s*(.*?)\s*(?:\[|$)', content, re.DOTALL)
        screenshot_section = re.search(r'\[Generated Screenshots\]\s*(.*?)\s*(?:\[|$)', content, re.DOTALL)
        video_paths = []
        if video_section:
            lines = video_section.group(1).strip().splitlines()
            for line in lines:
                line = line.strip()
                if line and os.path.isfile(line):
                    video_paths.append(os.path.normpath(line))
        screenshot_paths = []
        if screenshot_section:
            lines = screenshot_section.group(1).strip().splitlines()
            for line in lines:
                line = line.strip()
                if line and os.path.isfile(line) and line.lower().endswith(IMG_EXTENSIONS):
                    screenshot_paths.append(os.path.normpath(line))
        return video_paths, screenshot_paths

    def _load_task_data(self, video_paths, screenshot_paths):
        if not video_paths and not screenshot_paths:
            QMessageBox.warning(self, "提示", "任务文件中未找到有效的视频或截图路径。")
            return
        added_videos = 0
        for vp in video_paths:
            if vp not in self.video_files:
                self.video_files.append(vp)
                self.list_widget.addItem(os.path.basename(vp))
                added_videos += 1
        if added_videos > 0 and self.current_video_index == -1:
            self.load_video(0)
        if screenshot_paths:
            items = []
            target_names = set()
            for img_path in screenshot_paths:
                filename = os.path.basename(img_path)
                match = re.search(r'^(.+)_frame(\d+)_(.+)\.(jpg|jpeg|png)$', filename, re.IGNORECASE)
                if match:
                    base_name = match.group(1)
                    frame_num = int(match.group(2))
                    target_name = match.group(3)
                    video_name = self.match_video_by_basename(base_name)
                else:
                    frame_num = 0
                    target_name = "未知"
                    video_name = "unknown"
                items.append((img_path, video_name, frame_num, target_name))
                target_names.add(target_name)
            self.image_model.set_items(items)
            self.create_filter_checkboxes(target_names)
        msg = f"已加载 {len(video_paths)} 个视频"
        if screenshot_paths:
            msg += f" 和 {len(screenshot_paths)} 张截图"
        msg += "。"
        QMessageBox.information(self, "成功", msg)

    def load_task_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择任务文件", "", "任务文件 (*.txt)")
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法读取任务文件：{str(e)}")
            return
        video_paths, screenshot_paths = self._parse_task_content(content)
        self._load_task_data(video_paths, screenshot_paths)

    def load_task_file_from_path(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法读取任务文件：{str(e)}")
            return
        video_paths, screenshot_paths = self._parse_task_content(content)
        self._load_task_data(video_paths, screenshot_paths)

    def clear_filter_checkboxes(self):
        while self.filter_layout.count():
            child = self.filter_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.target_checkboxes.clear()

    def create_filter_checkboxes(self, target_names):
        self.clear_filter_checkboxes()
        for target in sorted(target_names):
            cb = QCheckBox(target)
            cb.setChecked(True)
            cb.stateChanged.connect(self.on_filter_changed)
            self.target_checkboxes[target] = cb
            self.filter_layout.addWidget(cb)

    def on_filter_changed(self):
        checked_targets = [t for t, cb in self.target_checkboxes.items() if cb.isChecked()]
        self.image_model.set_allowed_targets(checked_targets)

    def load_images_from_folder(self, folder_path=None):
        if not folder_path:
            folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder_path:
            return
        file_list = []
        target_names = set()
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(IMG_EXTENSIONS):
                file_path = os.path.normpath(os.path.join(folder_path, filename))
                if not os.path.isfile(file_path):
                    continue
                frame_match = re.search(r"_frame(\d+)_", filename)
                frame_num = int(frame_match.group(1)) if frame_match else 0
                base_match = re.search(r'^(.+)_frame\d+_', filename)
                if base_match:
                    base_name = base_match.group(1)
                    video_name = self.match_video_by_basename(base_name)
                else:
                    video_name = "unknown"
                target_match = re.search(r"_frame\d+_(.+)\.(jpg|jpeg|png)$", filename, re.IGNORECASE)
                target_name = target_match.group(1) if target_match else "未知"
                file_list.append((filename, file_path, video_name, frame_num, target_name))
                target_names.add(target_name)
        if not file_list:
            QMessageBox.information(self, "提示", "文件夹中未找到图片文件")
            return
        def natural_sort_key(item):
            filename = item[0]
            parts = re.split(r'(\d+)', filename.lower())
            return [int(part) if part.isdigit() else part for part in parts]
        file_list.sort(key=natural_sort_key)
        items = [(fp, vn, fn, tn) for _, fp, vn, fn, tn in file_list]
        self.image_model.set_items(items)
        self.create_filter_checkboxes(target_names)

    def on_image_clicked(self, index):
        item = self.image_model.get_item(index)
        if not item:
            return
        img_path, video_name, frame_num, target_name = item
        start_frame = max(0, frame_num - 30)
        video_idx = -1
        for i, path in enumerate(self.video_files):
            if os.path.basename(path) == video_name:
                video_idx = i
                break
        if video_idx == -1:
            video_base = os.path.splitext(video_name)[0]
            for i, path in enumerate(self.video_files):
                if os.path.splitext(os.path.basename(path))[0] == video_base:
                    video_idx = i
                    break
        if video_idx == -1:
            similar_videos = [
                os.path.basename(p) for p in self.video_files
                if video_base.lower() in os.path.splitext(os.path.basename(p))[0].lower()
            ]
            if similar_videos:
                msg = (f"未找到精确匹配的视频 '{video_name}'\n"
                       f"找到相似视频（基础名包含 '{video_base}'）：\n" +
                       "\n".join(f"• {v}" for v in similar_videos[:5]))
                QMessageBox.warning(self, "视频未找到", msg)
            else:
                QMessageBox.warning(self, "视频未找到",
                                    f"未找到与截图关联的视频：'{video_name}'\n"
                                    f"请确保已加载对应视频文件（基础名：{video_base})\n"
                                    f"支持格式：MP4/AVI/MOV/MKV/WMV/FLV/WEBM 等")
            return
        self.load_video(video_idx)
        self.current_frame = start_frame
        self.show_current_frame()
        self.timeline.set_current_frame(start_frame)
        self.play_forward()

    def capture_screenshot(self):
        if not self.cap or self.current_frame < 0 or self.current_video_index == -1:
            QMessageBox.warning(self, "提示", "当前无可用视频帧，无法截图")
            return
        screenshot_dir = os.path.join(os.getcwd(), "screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)
        video_basename = os.path.splitext(os.path.basename(self.video_files[self.current_video_index]))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        screenshot_filename = f"{video_basename}_frame{self.current_frame}_{timestamp}.jpg"
        screenshot_path = os.path.normpath(os.path.join(screenshot_dir, screenshot_filename))
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                QMessageBox.warning(self, "提示", "读取当前视频帧失败，无法截图")
                return
            success, img_encoded = cv2.imencode('.jpg', frame)
            if success:
                with open(screenshot_path, 'wb') as f:
                    f.write(img_encoded.tobytes())
                QMessageBox.information(self, "截图成功", f"截图已保存至：\n{screenshot_path}")
            else:
                QMessageBox.warning(self, "提示", "图片编码失败，无法保存截图")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存截图失败：{str(e)[:50]}")
            print(f"截图保存异常：{e}")

    def get_next_speed(self, current, direction):
        if direction == "up":
            for s in DEFAULT_PLAYBACK_SPEEDS:
                if s > current:
                    return s
            return DEFAULT_PLAYBACK_SPEEDS[-1]
        else:
            for s in reversed(DEFAULT_PLAYBACK_SPEEDS):
                if s < current:
                    return s
            return DEFAULT_PLAYBACK_SPEEDS[0]

    def set_playback_speed(self, speed):
        self.playback_speed = speed
        if speed == int(speed):
            self.speed_label.setText(f"{int(speed)}x")
        else:
            self.speed_label.setText(f"{speed:.2g}x")
        if self.is_playing:
            self.restart_timer()

    def increase_speed(self):
        new_speed = self.get_next_speed(self.playback_speed, "up")
        self.set_playback_speed(new_speed)

    def decrease_speed(self):
        new_speed = self.get_next_speed(self.playback_speed, "down")
        self.set_playback_speed(new_speed)

    def _save_play_state(self):
        self.was_playing_before_slider = self.is_playing
        if self.is_playing:
            self.pause()

    def timeline_changed(self, frame):
        self.current_frame = frame
        self.show_current_frame()

    def timeline_released(self, frame):
        self.current_frame = frame
        self.show_current_frame()
        if self.was_playing_before_slider:
            self.is_playing = True
            self.restart_timer()

    def format_time(self, seconds):
        if seconds < 0: seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def update_frame_info(self):
        total_time = self.total_frames / self.fps if self.fps > 0 else 0
        current_time = self.current_frame / self.fps if self.fps > 0 else 0
        frame_str = f"{self.current_frame} / {self.total_frames}"
        time_str = f"{self.format_time(current_time)} / {self.format_time(total_time)}"
        self.frame_info_label.setText(f"{frame_str}    |    {time_str}")

    def add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择视频文件", "", f"视频文件 (*{' *'.join(VIDEO_EXTENSIONS)})"
        )
        for f in files:
            if f not in self.video_files:
                self.video_files.append(f)
                self.list_widget.addItem(os.path.basename(f))
        if self.video_files and self.current_video_index == -1:
            self.load_video(0)

    def on_video_selected(self, item):
        self.load_video(self.list_widget.row(item))

    def load_video(self, index):
        self.pause()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_video_index = index
        path = self.video_files[index]
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", f"无法打开视频：{path}")
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = max(1.0, self.cap.get(cv2.CAP_PROP_FPS))
        self.current_frame = 0
        self.frame_input.setMaximum(self.total_frames - 1)
        self.file_input.setText(os.path.basename(path))
        self.timeline.set_total_frames(self.total_frames)
        self.timeline.set_fps(self.fps)
        self.timeline.set_current_frame(0)
        self.update_frame_info()
        self.show_current_frame()

    def restart_timer(self):
        if self.is_playing:
            interval_ms = max(10, int(1000 / (self.fps * self.playback_speed)))
            self.timer.start(interval_ms)

    def play_forward(self):
        if not self.cap: return
        self.is_playing = True
        self.play_direction = 1
        self.btn_play.setText("播放中")
        self.restart_timer()

    def play_reverse(self):
        if not self.cap: return
        self.is_playing = True
        self.play_direction = -1
        self.btn_play.setText("倒放中")
        self.restart_timer()

    def pause(self):
        self.is_playing = False
        self.timer.stop()
        self.btn_play.setText("播放")

    def step_forward(self):
        if not self.cap or self.current_frame >= self.total_frames - 1:
            return
        self.current_frame += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.show_current_frame()
        self.timeline.set_current_frame(self.current_frame)

    def step_backward(self):
        if not self.cap or self.current_frame <= 0:
            return
        self.current_frame -= 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.show_current_frame()
        self.timeline.set_current_frame(self.current_frame)

    def update_frame(self):
        if not self.cap or not self.is_playing:
            return
        if self.play_direction == 1:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 1
                self.display_frame(frame)
                self.timeline.set_current_frame(self.current_frame)
                self.update_frame_info()
            else:
                self.pause()
        else:
            next_frame = self.current_frame - 1
            if next_frame < 0:
                self.pause()
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = next_frame
                self.display_frame(frame)
                self.timeline.set_current_frame(self.current_frame)
                self.update_frame_info()
            else:
                self.pause()

    def show_current_frame(self):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.frame_input.setValue(self.current_frame)
            self.update_frame_info()
        else:
            self.video_label.clear()

    def display_frame(self, frame):
        h, w = frame.shape[:2]
        label_size = self.video_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            new_w, new_h = w, h
        else:
            scale_w = label_size.width() / w
            scale_h = label_size.height() / h
            scale = min(scale_w, scale_h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def jump_to_frame(self):
        target_file = self.file_input.text().strip()
        target_frame = self.frame_input.value()
        idx = -1
        for i, path in enumerate(self.video_files):
            if os.path.basename(path) == target_file:
                idx = i
                break
        if idx == -1:
            QMessageBox.warning(self, "未找到", f"视频列表中没有文件：{target_file}")
            return
        if idx != self.current_video_index:
            self.load_video(idx)
        if 0 <= target_frame < self.total_frames:
            self.current_frame = target_frame
            self.show_current_frame()
            self.timeline.set_current_frame(self.current_frame)
            self.pause()
        else:
            QMessageBox.warning(self, "帧号无效", f"帧号必须在 0 到 {self.total_frames - 1} 之间")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
            self.cap = None
        event.accept()

class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._default_text = "请加载视频"
        # 缓存绘图对象
        self._pen_default = QPen(COLOR_TEXT_DEFAULT)
        self._font_default = QFont()
        self._font_default.setPointSize(14)

    def sizeHint(self):
        return QSize(800, 600)

    def minimumSizeHint(self):
        return QSize(100, 80)

    def paintEvent(self, event):
        if not self.pixmap():
            painter = QPainter(self)
            painter.fillRect(self.rect(), COLOR_BG_DEFAULT)
            painter.setPen(self._pen_default)
            painter.setFont(self._font_default)
            painter.drawText(self.rect(), Qt.AlignCenter, self._default_text)
        else:
            super().paintEvent(event)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="任务文件路径 (.txt)")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.showMaximized()
    if args.task and os.path.isfile(args.task):
        QTimer.singleShot(100, lambda: player.load_task_file_from_path(args.task))
    sys.exit(app.exec())