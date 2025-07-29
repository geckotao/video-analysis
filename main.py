import ttkbootstrap as ttk  # 导入ttkbootstrap库，用于创建美观的GUI界面
import sys  # 导入sys模块，用于系统级操作
import os  # 导入os模块，用于文件和目录操作
import cv2  # 导入OpenCV库，用于视频处理和图像操作
import numpy as np  # 导入numpy库，用于数值计算和数组操作
from ultralytics import YOLO  # 导入YOLO模型，用于目标检测
import time  # 导入time模块，用于时间相关操作
import logging  # 导入logging模块，用于日志记录
from threading import Lock, Thread  # 导入线程相关模块，用于多线程处理
from queue import Queue  # 导入队列，用于线程间通信
import queue  # 导入queue模块，用于处理队列异常
from tkinter import filedialog, messagebox, Scale, HORIZONTAL  # 导入tkinter相关组件
import configparser  # 导入配置文件解析器
import concurrent.futures  # 导入并发执行模块，用于线程池管理
from ttkbootstrap.constants import *  # 导入ttkbootstrap常量

# --- 全局常量 (将在 __main__ 中初始化) ---
BATCH_SIZE = 2  # 帧处理批次大小
FRAME_QUEUE_SIZE = 15  # 帧队列最大容量
RESULT_QUEUE_SIZE = 15  # 结果队列最大容量
target_moving = 5  # 目标移动判定阈值
DEFAULT_CONFIDENCE_THRESHOLD = 0.1  # 默认置信度阈值

# --- 配置读取函数 ---
def get_config_value(config, section, option, default):
    """
    从配置对象中读取值并转换为与默认值相同的类型，读取失败则返回默认值
    
    参数:
        config: 配置解析器对象
        section: 配置节名
        option: 配置项名
        default: 默认值（用于确定返回值类型）
    
    返回:
        读取到的配置值或默认值
    """
    try:
        value = config.get(section, option)
        # 根据默认值类型进行转换
        if isinstance(default, int):
            return int(value)
        elif isinstance(default, float):
            return float(value)
        return value
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        # 记录日志，不直接弹窗
        logging.debug(f"读取配置 {section}.{option} 失败，使用默认值 {default}")
        return default

# --- 目标判断函数 ---
def is_target_moving(prev_positions, object_key, current_position, threshold):
    """
    检查目标是否移动超过指定阈值
    
    参数:
        prev_positions: 存储目标历史位置的字典
        object_key: 目标唯一标识
        current_position: 当前目标位置坐标 (x1, y1, x2, y2)
        threshold: 移动判定阈值
    
    返回:
        布尔值: 目标移动超过阈值返回True，否则返回False
    """
    prev_pos = prev_positions.get(object_key)
    if not prev_pos:  # 首次出现的目标视为移动
        return True
    # 检查任意坐标分量的移动是否超过阈值
    return any(abs(curr - prev) >= threshold for curr, prev in zip(current_position, prev_pos))

def is_target_captured(captured_targets, target_identifier):
    """
    检查目标是否已被捕获（去重判断）
    
    参数:
        captured_targets: 已捕获目标的集合
        target_identifier: 目标唯一标识符
    
    返回:
        布尔值: 已捕获返回True，否则返回False
    """
    return target_identifier in captured_targets

# --- 文件操作函数 ---
def save_detection_screenshot(original_frame, save_path, video_filename, frame_number, object_type, saved_objects_count, confidence):
    """
    保存检测到的目标截图
    
    参数:
        original_frame: 原始帧图像
        save_path: 保存路径
        video_filename: 视频文件名
        frame_number: 帧编号
        object_type: 目标类型名称
        saved_objects_count: 已保存目标计数
        confidence: 检测置信度
    
    返回:
        布尔值: 保存成功返回True，否则返回False
    """
    # 构建保存文件名
    file_name = os.path.join(save_path,
                             f"{os.path.splitext(video_filename)[0]}_{frame_number}_{object_type}_{saved_objects_count}_{confidence:.2f}.jpg")
    try:
        cv2.imwrite(file_name, original_frame)
        return True
    except Exception as e:
        logging.error(f"保存截图时出错: {e}")
        return False

def show_error(message):
    """
    显示错误信息并记录日志
    
    参数:
        message: 错误信息内容
    """
    messagebox.showerror("错误", message)
    logging.error(message)

# --- 视频捕获上下文管理器 ---
class VideoCaptureContextManager:
    """视频捕获上下文管理器，确保资源正确释放"""
    
    def __init__(self, video_path):
        """初始化上下文管理器，存储视频路径"""
        self.video_path = video_path
        self.cap = None  # 视频捕获对象

    def __enter__(self):
        """进入上下文，打开视频文件"""
        self.cap = cv2.VideoCapture(self.video_path)
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，释放视频资源"""
        if self.cap is not None:
            self.cap.release()

# --- 视频处理核心 ---
class VideoProcessingCore:
    """视频处理核心类，负责视频帧读取、目标检测和结果处理"""
    
    def __init__(self, progress_callback=None):
        """初始化视频处理核心"""
        self.paused = False  # 暂停状态标志
        self.stopped = False  # 停止状态标志
        self.frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)  # 帧队列
        self.result_queue = Queue(maxsize=RESULT_QUEUE_SIZE)  # 结果队列（用于同步）
        self.lock = Lock()  # 通用锁
        self.pause_stop_lock = Lock()  # 暂停/停止状态锁
        self.last_screenshot_time = {}  # 最后截图时间记录
        self.last_screenshot = {}  # 最后截图记录
        self.roi_points = []  # ROI区域点集合
        self.roi_mask = None  # ROI掩码
        self.progress_callback = progress_callback  # 进度回调函数
        
        # 状态变量，每次处理新视频前需要重置
        self.saved_objects = {}  # 已保存目标记录
        self.detection_info = []  # 检测信息列表
        self.prev_positions = {}  # 目标历史位置记录
        self.captured_targets = set()  # 已捕获目标集合

    def reset(self):
        """重置处理状态，为处理新视频做准备"""
        logging.debug("重置 VideoProcessingCore 状态")
        # 清空各类状态存储
        self.saved_objects.clear()
        self.detection_info.clear()
        self.prev_positions.clear()
        self.captured_targets.clear()
        self.last_screenshot_time.clear()
        self.last_screenshot.clear()
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        # 不重置ROI，因为它可能在多个视频间共享

    def detect_and_save(self, frames, original_frames, model, target_classes, class_mapping, current_times, save_path,
                        only_movement_targets, video_filename, fps, confidence_threshold):
        """
        执行模型推理并保存符合条件的截图
        
        参数:
            frames: 待检测的帧列表（可能应用了ROI）
            original_frames: 原始帧列表（未应用ROI，用于保存）
            model: YOLO检测模型
            target_classes: 目标类别ID列表
            class_mapping: 类别ID到名称的映射
            current_times: 各帧对应的时间戳
            save_path: 截图保存路径
            only_movement_targets: 是否只保存移动目标
            video_filename: 视频文件名
            fps: 视频帧率
            confidence_threshold: 置信度阈值
        
        返回:
            检测结果列表或None（出错时）
        """
        try:
            results = model(frames)  # 模型推理
            
            for i, result in enumerate(results):
                boxes = result.boxes  # 获取检测框
                current_time = current_times[i]  # 当前帧时间戳
                original_frame = original_frames[i]  # 原始帧
                
                for box in boxes:
                    cls_id = int(box.cls[0])  # 类别ID
                    confidence = float(box.conf[0])  # 置信度
                    
                    # 筛选符合条件的目标（目标类别和置信度）
                    if cls_id in target_classes and confidence >= confidence_threshold:
                        # 获取目标边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        object_key = f"{cls_id}_{x1}_{y1}_{x2}_{y2}"  # 目标唯一标识
                        object_type = class_mapping[cls_id]  # 目标类型名称
                        
                        # 移动检测（如果启用）
                        if only_movement_targets and not is_target_moving(
                                self.prev_positions, object_key, (x1, y1, x2, y2), target_moving):
                            continue  # 非移动目标跳过
                        
                        self.prev_positions[object_key] = (x1, y1, x2, y2)  # 更新目标位置
                        
                        # 目标去重处理（坐标粗略量化，减少重复）
                        target_identifier = f"{cls_id}_{round(x1 / 10)}_{round(y1 / 10)}_{round(x2 / 10)}_{round(y2 / 10)}"
                        if is_target_captured(self.captured_targets, target_identifier):
                            continue  # 已捕获目标跳过
                        
                        # 保存截图
                        if object_key not in self.saved_objects:
                            # 记录检测信息
                            info = f"{object_type} 在 {current_time:.2f} 秒时出现在位置: ({x1}, {y1}), ({x2}, {y2})，置信度: {confidence:.2f}"
                            self.detection_info.append(info)
                            
                            frame_number = int(current_time * fps)  # 计算帧编号
                            # 保存截图
                            if save_detection_screenshot(original_frame, save_path, video_filename, frame_number, 
                                                        object_type, len(self.saved_objects), confidence):
                                self.saved_objects[object_key] = True
                                self.captured_targets.add(target_identifier)
                                self.last_screenshot_time[object_type] = current_time
                                self.last_screenshot[object_type] = original_frame.copy()
            
            return results
        except Exception as e:
            show_error(f"模型推理出错: {e}")
            return None

    def read_frames(self, video_path, speed_multiplier):
        """
        读取视频帧并放入队列
        
        参数:
            video_path: 视频文件路径
            speed_multiplier: 速度倍率（控制跳帧读取）
        """
        logging.info(f"开始读取视频: {video_path}")
        with VideoCaptureContextManager(video_path) as cap:
            if not cap.isOpened():
                show_error("无法打开视频文件")
                self.frame_queue.put(None)  # 通知处理线程结束
                return
            
            frame_count = 0  # 已读取帧计数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
            fps = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率
            video_filename = os.path.basename(video_path)  # 视频文件名
            
            try:
                while True:
                    # 检查暂停/停止状态
                    with self.pause_stop_lock:
                        if self.stopped:
                            logging.info("读取帧线程被停止")
                            break
                        if self.paused:
                            time.sleep(0.1)
                            continue
                    
                    frames = []  # 当前批次帧（处理用）
                    original_frames = []  # 当前批次原始帧（保存用）
                    current_times = []  # 当前批次帧时间戳
                    
                    # 按批次读取帧
                    for _ in range(BATCH_SIZE):
                        # 根据速度倍率跳帧
                        for _ in range(int(speed_multiplier)):
                            if not cap.grab():  # 抓取帧（不解码）
                                break
                        
                        ret, frame = cap.retrieve()  # 解码并获取帧
                        if ret:
                            current_time = frame_count / fps  # 计算时间戳
                            frames.append(frame)
                            original_frames.append(frame)
                            current_times.append(current_time)
                            frame_count += int(speed_multiplier)  # 更新帧计数
                            
                            # 更新读取进度（0% - 50%）
                            if self.progress_callback and total_frames > 0:
                                read_progress = (frame_count / total_frames) * 50.0
                                self.progress_callback(min(read_progress, 50.0))  # 不超过50%
                        else:
                            break  # 读取失败，退出循环
                    
                    # 将批次帧放入队列
                    if frames:
                        try:
                            self.frame_queue.put(
                                (frames, original_frames, current_times, video_filename, fps, total_frames), 
                                timeout=1
                            )
                        except queue.Full:
                            logging.warning("帧队列已满，跳过当前帧批次")
                            continue
                    else:
                        break  # 没有帧可处理，退出循环
            
            except Exception as e:
                show_error(f"读取帧时出错: {e}")
            finally:
                self.frame_queue.put(None)  # 通知处理线程结束
        
        logging.info(f"视频读取结束: {video_path}")

    def process_frames(self, model, target_classes, class_mapping, save_path, only_movement_targets, confidence_threshold):
        """
        从队列获取帧，应用ROI，执行推理并保存结果
        
        参数:
            model: YOLO检测模型
            target_classes: 目标类别ID列表
            class_mapping: 类别ID到名称的映射
            save_path: 截图保存路径
            only_movement_targets: 是否只保存移动目标
            confidence_threshold: 置信度阈值
        """
        logging.info("开始处理帧")
        processed_count = 0  # 已处理帧计数
        total_frames = 0  # 总帧数
        
        # 获取速度倍率（确保有效）
        speed_multiplier = getattr(self, 'speed_multiplier', 1)
        speed_multiplier = max(1, speed_multiplier)

        try:
            while True:
                item = self.frame_queue.get()  # 从队列获取帧数据
                if item is None:
                    break  # 结束标志
                
                # 解包队列数据（兼容不同格式）
                if len(item) == 6:
                    frames, original_frames, current_times, video_filename, fps, total_frames_from_queue = item
                elif len(item) == 5:
                    frames, original_frames, current_times, video_filename, fps = item
                    total_frames_from_queue = 0
                else:
                    logging.error(f"Process_frames 从队列获取到意外的数据项大小: {len(item)}")
                    continue
                
                # 更新总帧数
                if total_frames <= 0 and total_frames_from_queue > 0:
                    total_frames = total_frames_from_queue
                    logging.debug(f"Process_frames 获取到总帧数: {total_frames}")
                
                # 应用ROI
                roi_frames = self.apply_roi(frames)
                
                # 执行检测和保存
                results = self.detect_and_save(
                    roi_frames, original_frames, model, target_classes, class_mapping,
                    current_times, save_path, only_movement_targets,
                    video_filename, fps, confidence_threshold
                )
                
                # 更新处理进度
                batch_size = len(frames)
                processed_count += batch_size * speed_multiplier  # 按速度倍率计算
                
                if self.progress_callback and total_frames > 0:
                    # 计算处理进度（50% - 100%）
                    processing_progress = 50.0 + (float(processed_count) / float(total_frames)) * 50.0
                    processing_progress = min(processing_progress, 100.0)  # 不超过100%
                    self.progress_callback(processing_progress)
                
                # 同步结果队列
                for i in range(len(original_frames)):
                    try:
                        self.result_queue.put(("PROCESSED", None), timeout=1)
                    except queue.Full:
                        logging.warning("结果队列已满，跳过当前结果同步")
                        continue
        
        except Exception as e:
            show_error(f"处理帧时出错: {e}")
        finally:
            # 确保进度更新到100%
            if self.progress_callback:
                self.progress_callback(100.0)
            self.result_queue.put(None)  # 通知结果处理结束
        
        logging.info("帧处理结束")

    def apply_roi(self, frames):
        """
        应用ROI掩码到帧
        
        参数:
            frames: 帧列表
        
        返回:
            应用ROI后的帧列表
        """
        if self.roi_points and len(frames) > 0:
            # 检查并创建/更新ROI掩码
            frame_shape = frames[0].shape[:2]
            if self.roi_mask is None or self.roi_mask.shape != frame_shape:
                logging.info("创建或更新ROI掩码")
                self.create_roi_mask(frame_shape)
            
            if self.roi_mask is not None:
                # 对每帧应用掩码
                return [cv2.bitwise_and(frame, frame, mask=self.roi_mask) for frame in frames]
        
        return frames  # 无ROI时返回原帧

    def create_roi_mask(self, frame_shape):
        """
        创建ROI掩码
        
        参数:
            frame_shape: 帧的形状 (高度, 宽度)
        """
        with self.lock:
            if not self.roi_points:
                self.roi_mask = None
                return
            
            # 创建空白掩码并填充ROI区域
            self.roi_mask = np.zeros(frame_shape, dtype=np.uint8)
            roi_pts = np.array([self.roi_points], dtype=np.int32)
            cv2.fillPoly(self.roi_mask, roi_pts, 255)  # 填充ROI区域为白色

    def display_results(self, completion_callback):
        """
        等待处理完成并调用回调（用于同步）
        
        参数:
            completion_callback: 处理完成后的回调函数
        """
        logging.info("开始等待处理完成")
        try:
            while True:
                item = self.result_queue.get()
                if item is None:
                    break  # 处理结束
                
                # 检查是否被停止
                with self.pause_stop_lock:
                    if self.stopped:
                        logging.info("显示线程检测到停止信号")
                        break
        
        except Exception as e:
            show_error(f"等待处理完成时出错: {e}")
        finally:
            # 确保回调被调用
            if completion_callback:
                completion_callback()
        
        logging.info("处理完成等待结束")

    def start_processing(self, video_path, model, target_classes, class_mapping, save_path, speed_multiplier,
                         only_movement_targets, completion_callback, confidence_threshold):
        """
        启动视频处理流程
        
        参数:
            video_path: 视频文件路径
            model: YOLO检测模型
            target_classes: 目标类别ID列表
            class_mapping: 类别ID到名称的映射
            save_path: 截图保存路径
            speed_multiplier: 速度倍率
            only_movement_targets: 是否只保存移动目标
            completion_callback: 处理完成后的回调函数
            confidence_threshold: 置信度阈值
        """
        self.reset()  # 重置状态
        self.speed_multiplier = max(1, speed_multiplier)  # 保存速度倍率并确保有效
        logging.info(f"启动处理流程 for {video_path}")
        
        def run_pipeline():
            """运行处理管道的内部函数"""
            try:
                # 使用线程池管理三个主要线程
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    # 提交帧读取任务
                    future_read = executor.submit(self.read_frames, video_path, speed_multiplier)
                    # 提交帧处理任务
                    future_process = executor.submit(self.process_frames, model, target_classes, class_mapping, 
                                                   save_path, only_movement_targets, confidence_threshold)
                    # 提交结果等待任务
                    future_display = executor.submit(self.display_results, completion_callback)
                    
                    # 等待所有任务完成
                    concurrent.futures.wait(
                        [future_read, future_process, future_display], 
                        return_when=concurrent.futures.ALL_COMPLETED
                    )
            
            except Exception as e:
                show_error(f"线程池执行出错: {e}")
            finally:
                logging.info(f"处理流程结束 for {video_path}")
        
        # 在新线程中运行整个管道，避免阻塞GUI
        processing_thread = Thread(target=run_pipeline)
        processing_thread.daemon = True  # 使线程随主程序退出
        processing_thread.start()

    def clear_roi(self):
        """清除ROI点和掩码"""
        with self.lock:
            self.roi_points = []
            self.roi_mask = None

# --- YOLO 类名 ---
yolo_classes = [
    "人", "自行车", "小车", "摩托车", "飞机", "巴士", "火车", "货车", "船", "交通灯", "消防栓",
    "停车标志", "停车计费器", "长椅", "鸟", "猫", "狗", "马", "羊", "牛", "大象",
    "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带", "行李箱", "飞盘", "滑雪板",
    "滑雪橇", "运动球", "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子", "酒杯",
    "茶杯", "叉子", "刀", "勺子", "碗", "香蕉", "苹果", "三明治", "橙子", "西兰花",
    "胡萝卜", "热狗", "披萨", "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽植物", "床", "餐桌",
    "马桶", "电视", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱", "烤面包机",
    "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷"
]  # COCO数据集类别中文名称列表

# --- ROI 辅助函数 ---
def is_polygon_closed(roi_points):
    """检查多边形是否闭合（至少3个点）"""
    return len(roi_points) >= 3

def is_polygon_non_crossing(roi_points):
    """
    检查多边形是否为简单多边形（边不交叉）
    
    参数:
        roi_points: 多边形顶点列表
    
    返回:
        布尔值: 非交叉返回True，否则返回False
    """
    def orientation(p, q, r):
        """计算三个点的方向（顺时针/逆时针/共线）"""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0  # 共线
        return 1 if val > 0 else 2  # 顺时针/逆时针

    def on_segment(p, q, r):
        """检查点q是否在p和r构成的线段上"""
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    def do_intersect(p1, q1, p2, q2):
        """检查线段p1q1和p2q2是否相交"""
        o1, o2, o3, o4 = (orientation(p1, q1, p2), orientation(p1, q1, q2),
                          orientation(p2, q2, p1), orientation(p2, q2, q1))
        
        # 一般情况：线段相交
        if o1 != o2 and o3 != o4: return True
        
        # 特殊情况：共线且重叠
        if (o1 == 0 and on_segment(p1, p2, q1)) or (o2 == 0 and on_segment(p1, q2, q1)) or \
           (o3 == 0 and on_segment(p2, p1, q2)) or (o4 == 0 and on_segment(p2, q1, q2)):
            return True
        
        return False

    n = len(roi_points)
    if n < 3: return True  # 少于3个点不构成多边形
    
    # 检查所有边是否相交
    for i in range(n):
        for j in range(i + 2, n):
            # 跳过相邻边（多边形相邻边自然相连，不算交叉）
            if (i == 0 and j == n - 1) or (i == n - 1 and j == 0):
                continue
            if do_intersect(roi_points[i], roi_points[(i + 1) % n], 
                           roi_points[j], roi_points[(j + 1) % n]):
                return False  # 检测到交叉
    
    return True

# --- GUI 类 ---
class VideoProcessorGUI:
    """视频处理器GUI类，负责用户界面和交互逻辑"""
    
    def __init__(self, root):
        """初始化GUI界面"""
        self.root = root  # 主窗口
        self.core = VideoProcessingCore(progress_callback=self.update_progress)  # 视频处理核心
        self.model = None  # YOLO模型
        self.video_paths = []  # 视频文件路径列表
        self.current_video_index = 0  # 当前处理视频索引
        self.create_widgets()  # 创建界面组件
        self.load_model()  # 加载YOLO模型

    def update_progress(self, progress):
        """更新进度条"""
        self.progress_bar['value'] = progress
        self.root.update_idletasks()  # 刷新界面

    def create_widgets(self):
        """创建GUI界面组件"""
        # 视频选择区域
        self.video_frame = self.create_frame("选择视频文件:")
        self.video_frame.pack(pady=2)
        self.video_entry = self.create_entry(self.video_frame)
        self.create_button(self.video_frame, "浏览", self.select_video, "success")
        
        # 保存路径选择区域
        self.save_path_frame = self.create_frame("截图保存路径:")
        self.save_path_frame.pack(pady=2)
        self.save_path_entry = self.create_entry(self.save_path_frame)
        self.create_button(self.save_path_frame, "浏览", self.select_save_path, "info")
        
        # 目标选择区域
        self.object_frame = ttk.LabelFrame(self.root, text="选择要检测的目标:")
        self.object_frame.pack(pady=2, fill=ttk.X, padx=5)
        self.checkbox_vars = {cls: ttk.BooleanVar() for cls in yolo_classes}  # 目标选择复选框变量
        
        # 使用网格布局排列目标选择复选框
        for i, cls in enumerate(yolo_classes):
            row = i // 7  # 每行7个复选框
            col = i % 7
            checkbox = ttk.Checkbutton(self.object_frame, text=cls, variable=self.checkbox_vars[cls])
            checkbox.grid(row=row, column=col, sticky=ttk.W, padx=2, pady=2)
        
        # 全选/取消全选按钮
        self.button_container_frame = ttk.Frame(self.object_frame)
        button_row = (len(yolo_classes) - 1) // 7 + 1
        self.button_container_frame.grid(row=button_row, column=0, columnspan=7, sticky=ttk.W, pady=5)
        self.select_button = self.create_button(self.button_container_frame, "全选", self.toggle_select_all, "primary")
        
        # 分析倍速选择区域
        self.speed_group_frame = ttk.LabelFrame(self.root, text="分析倍速 (丢帧实现，请慎用)")
        self.speed_group_frame.pack(pady=5, fill=ttk.X, padx=5)
        self.speed_frame = ttk.Frame(self.speed_group_frame)
        self.speed_frame.pack(pady=2)
        self.speed_var = ttk.IntVar(value=1)  # 倍速变量
        self.create_label(self.speed_frame, "倍速:")
        speeds = [1, 2, 4, 8, 16, 24, 32, 48, 64]  # 可选倍速
        for speed in speeds:
            radio = ttk.Radiobutton(self.speed_frame, text=f"{speed}X", variable=self.speed_var, value=speed)
            radio.pack(side=ttk.LEFT, padx=2)
        
        # ROI选择区域
        self.roi_frame = ttk.Frame(self.root)
        self.roi_frame.pack(pady=2)
        self.roi_button = self.create_button(self.roi_frame, "选择关注区域", self.select_roi, "warning")
        self.roi_status_label = self.create_label(self.roi_frame, "未选取关注区域", anchor=ttk.CENTER)
        
        # 控制按钮区域
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=2)
        self.start_button = self.create_button(self.button_frame, "开始处理", self.start_processing, "primary")
        self.pause_button = self.create_button(self.button_frame, "暂停处理", self.pause_processing, "secondary")
        self.pause_button.config(state=ttk.DISABLED)  # 初始禁用暂停按钮
        self.create_button(self.button_frame, "结束处理", self.stop_processing, "danger")
        
        # 移动目标选项
        self.movement_frame = ttk.Frame(self.root)
        self.movement_frame.pack(pady=2)
        self.only_movement_var = ttk.BooleanVar(value=True)  # 只检测移动目标变量
        checkbox = ttk.Checkbutton(self.movement_frame, text="只对活动目标截图", variable=self.only_movement_var)
        checkbox.pack(side=ttk.LEFT, padx=5)
        
        # 置信度阈值调节
        self.confidence_frame = ttk.LabelFrame(self.root, text="置信度阈值")
        self.confidence_frame.pack(pady=5, fill=ttk.X, padx=5)
        self.confidence_var = ttk.DoubleVar(value=DEFAULT_CONFIDENCE_THRESHOLD)  # 置信度变量
        self.confidence_scale = Scale(self.confidence_frame, from_=0.0, to=1.0, resolution=0.01,
                                      orient=HORIZONTAL, variable=self.confidence_var)
        self.confidence_scale.pack(side=ttk.LEFT, fill=ttk.X, expand=True, padx=5)
        self.confidence_label = self.create_label(self.confidence_frame, f"{self.confidence_var.get():.2f}")
        self.confidence_scale.config(command=self.update_confidence_label)  # 绑定值变化事件
        self.confidence_label.pack(side=ttk.LEFT, padx=5)
        
        # 注意事项
        self.note_frame = ttk.Frame(self.root)
        self.note_frame.pack(pady=2)
        self.create_label(self.note_frame, "注意：分析过程中不要关闭程序，在弹出【视频分析已结束】窗口后再关闭。", anchor=ttk.CENTER)
        
        # 进度条区域
        self.progress_frame = ttk.Frame(self.root)
        self.progress_frame.pack(pady=0, padx=5, fill=ttk.X)
        self.progress_label = ttk.Label(self.progress_frame, text="等待处理视频")
        self.progress_label.pack(pady=5, fill=ttk.X)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=ttk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(pady=5, fill=ttk.X, expand=True)

    def update_confidence_label(self, value):
        """更新置信度显示标签"""
        self.confidence_label.config(text=f"{float(value):.2f}")

    def create_frame(self, label_text):
        """创建带标签的框架"""
        frame = ttk.Frame(self.root)
        self.create_label(frame, label_text)
        return frame

    def create_entry(self, frame):
        """创建输入框"""
        entry = ttk.Entry(frame, width=50)
        entry.pack(side=ttk.LEFT, padx=5)
        return entry

    def create_button(self, frame, text, command, bootstyle):
        """创建按钮"""
        button = ttk.Button(frame, text=text, command=command, bootstyle=bootstyle)
        button.pack(side=ttk.LEFT, padx=5)
        return button

    def create_label(self, frame, text, anchor=None):
        """创建标签"""
        label = ttk.Label(frame, text=text)
        if anchor:
            label.pack(anchor=anchor)
        else:
            label.pack(side=ttk.LEFT, padx=5)
        return label

    def select_video(self):
        """选择视频文件"""
        try:
            self.video_paths = filedialog.askopenfilenames(
                filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv")]
            )
            if self.video_paths:
                video_paths_str = ", ".join(self.video_paths)
                self.video_entry.delete(0, ttk.END)
                self.video_entry.insert(0, video_paths_str)
        except Exception as e:
            show_error(f"选择视频文件时出错: {e}")

    def select_save_path(self):
        """选择截图保存路径"""
        try:
            save_path = filedialog.askdirectory()
            if save_path:
                self.save_path_entry.delete(0, ttk.END)
                self.save_path_entry.insert(0, save_path)
        except Exception as e:
            show_error(f"选择保存路径时出错: {e}")

    def get_selected_classes(self):
        """获取选中的目标类别"""
        return [text for text, var in self.checkbox_vars.items() if var.get()]

    def load_model(self):
        """加载YOLO模型"""
        try:
            # 从配置文件获取模型路径
            model_path = get_config_value(config, 'Parameters', 'model_path', 'models/yolo11x.pt')
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                # 尝试在脚本目录下查找
                script_dir_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
                if os.path.exists(script_dir_model_path):
                    model_path = script_dir_model_path
                else:
                    show_error(f"模型文件未找到: {model_path}。请检查配置文件或模型路径。")
                    return
            
            self.model = YOLO(model_path)
            logging.info(f"模型加载成功: {model_path}")
        except Exception as e:
            show_error(f"加载模型时出错: {e}")

    def start_processing(self):
        """开始处理视频"""
        logging.info("开始处理视频")
        save_path = self.save_path_entry.get().strip()
        selected_classes = self.get_selected_classes()
        speed_multiplier = self.speed_var.get()
        only_movement_targets = self.only_movement_var.get()
        confidence_threshold = self.confidence_var.get()

        # 输入验证
        if not self.video_paths or not save_path:
            show_error("请选择视频文件和截图保存路径")
            return
        if not selected_classes:
            show_error("请选择至少一个检测对象")
            return
        
        # 创建保存目录（如果不存在）
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
                logging.info(f"创建保存路径: {save_path}")
            except Exception as e:
                show_error(f"创建保存路径时出错: {e}")
                return
        
        # 检查模型是否加载
        if self.model is None:
            show_error("模型未加载，请检查模型文件是否存在")
            return

        # 构建类别映射和目标类别ID列表
        class_mapping = {i: name for i, name in enumerate(yolo_classes)}
        target_classes = [key for key, value in class_mapping.items() if value in selected_classes]

        # 更新UI状态
        self.start_button.config(state=ttk.DISABLED)
        self.roi_button.config(state=ttk.DISABLED)
        self.pause_button.config(state=ttk.NORMAL, text="暂停处理")
        self.current_video_index = 0

        # 重置处理状态
        with self.core.pause_stop_lock:
            self.core.paused = False
            self.core.stopped = False

        # 开始处理视频
        self.process_next_video(save_path, target_classes, class_mapping, speed_multiplier, 
                               only_movement_targets, confidence_threshold)

    def process_next_video(self, save_path, target_classes, class_mapping, speed_multiplier, 
                          only_movement_targets, confidence_threshold):
        """处理下一个视频（递归调用）"""
        if self.current_video_index < len(self.video_paths):
            video_path = self.video_paths[self.current_video_index]
            video_filename = os.path.basename(video_path)
            # 更新进度标签
            self.progress_label.config(
                text=f"正在处理: {video_filename} ({self.current_video_index + 1}/{len(self.video_paths)})"
            )
            logging.info(f"开始处理视频: {video_path}")

            # 定义处理完成后的回调
            def on_video_processed():
                logging.info(f"视频处理完成回调: {video_path}")
                self.current_video_index += 1
                # 在主线程中调度下一个视频的处理
                self.root.after(10, lambda: self.process_next_video(
                    save_path, target_classes, class_mapping, speed_multiplier, 
                    only_movement_targets, confidence_threshold
                ))

            # 启动核心处理逻辑
            self.core.start_processing(
                video_path, self.model, target_classes, class_mapping,
                save_path, speed_multiplier, only_movement_targets,
                on_video_processed, confidence_threshold
            )
        else:
            self.on_all_videos_processed()  # 所有视频处理完成

    def on_all_videos_processed(self):
        """所有视频处理完成后的操作"""
        logging.info("所有视频分析已结束")
        # 恢复UI状态
        self.start_button.config(state=ttk.NORMAL)
        self.roi_button.config(state=ttk.NORMAL)
        self.pause_button.config(state=ttk.DISABLED)
        self.progress_bar['value'] = 0
        self.progress_label.config(text="等待处理视频")
        messagebox.showinfo("提示", f"所有视频分析已结束！")

    def pause_processing(self):
        """暂停/继续处理"""
        with self.core.pause_stop_lock:
            self.core.paused = not self.core.paused
        
        # 更新按钮文本
        if self.core.paused:
            self.pause_button.config(text="继续处理")
            logging.info("处理已暂停")
        else:
            self.pause_button.config(text="暂停处理")
            logging.info("处理已继续")

    def stop_processing(self):
        """停止处理"""
        with self.core.pause_stop_lock:
            self.core.stopped = True
            self.core.paused = False
        
        # 恢复UI状态
        self.pause_button.config(state=ttk.DISABLED)
        self.start_button.config(state=ttk.NORMAL)
        self.roi_button.config(state=ttk.NORMAL)
        self.current_video_index = 0
        self.progress_bar['value'] = 0
        self.progress_label.config(text="处理已停止")
        logging.info("处理已停止")

    def toggle_select_all(self):
        """全选/取消全选目标类别"""
        all_selected = all(var.get() for var in self.checkbox_vars.values())
        if all_selected:
            # 取消全选
            for var in self.checkbox_vars.values():
                var.set(False)
            self.select_button.config(text="全选")
        else:
            # 全选
            for var in self.checkbox_vars.values():
                var.set(True)
            self.select_button.config(text="取消全选")

    # --- ROI 选择相关方法 ---
    def _update_roi_preview(self, base_frame, points, out_frame):
        """更新ROI选择预览图像"""
        out_frame[:] = base_frame[:]  # 复制原始帧
        
        # 绘制ROI点和连线
        for i, p in enumerate(points):
            cv2.circle(out_frame, p, 5, (0, 0, 255), -1)  # 绘制点
            if i > 0:
                cv2.line(out_frame, points[i - 1], p, (0, 255, 0), 2)  # 绘制线段
        
        # 闭合多边形（如果有多个点）
        if len(points) > 1:
            cv2.line(out_frame, points[-1], points[0], (0, 255, 0), 2)

    def select_roi(self):
        """选择ROI区域（关注区域）"""
        try:
            if not self.video_paths:
                show_error("请先选择视频文件")
                return
            
            # 初始化ROI选择所需变量
            self._roi_points = []  # 存储ROI点
            self._roi_frame = None  # 存储用于ROI选择的帧
            
            # 从第一个视频读取一帧用于ROI选择
            cap = cv2.VideoCapture(self.video_paths[0])
            ret, frame = cap.read()
            cap.release()
            if not ret:
                show_error("无法读取视频文件以选择ROI")
                return
            
            self._roi_frame = frame.copy()  # 保存原始帧
            temp_frame = self._roi_frame.copy()  # 用于显示的临时帧
            
            # 创建ROI选择窗口
            cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Select ROI", 800, 600)

            def mouse_callback(event, x, y, flags, param):
                """鼠标回调函数，处理ROI点选择"""
                if event == cv2.EVENT_LBUTTONDOWN:
                    # 左键点击添加点
                    self._roi_points.append((x, y))
                    self._update_roi_preview(self._roi_frame, self._roi_points, temp_frame)
                    cv2.imshow("Select ROI", temp_frame)
                elif event == cv2.EVENT_RBUTTONDOWN and self._roi_points:
                    # 右键点击删除最后一个点
                    self._roi_points.pop()
                    self._update_roi_preview(self._roi_frame, self._roi_points, temp_frame)
                    cv2.imshow("Select ROI", temp_frame)

            # 设置鼠标回调
            cv2.setMouseCallback("Select ROI", mouse_callback)
            cv2.imshow("Select ROI", temp_frame)  # 初始显示

            # ROI选择主循环
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    # 重置ROI选择
                    self._roi_points = []
                    temp_frame = self._roi_frame.copy()
                    self._update_roi_preview(self._roi_frame, self._roi_points, temp_frame)
                    cv2.imshow("Select ROI", temp_frame)
                elif key == ord('c') or key == 13:  # 'c'键或回车键确认
                    if self._roi_points:
                        # 验证ROI是否合法
                        if is_polygon_closed(self._roi_points) and is_polygon_non_crossing(self._roi_points):
                            self.core.roi_points = self._roi_points  # 应用ROI
                            self.roi_status_label.config(text="已选取关注区域")
                            self.roi_button.config(text="取消选取", command=self.cancel_roi)
                            logging.info("ROI 选取成功")
                        else:
                            show_error("选取的关注区域不合法，请确保区域至少有3个点且不交叉。")
                    cv2.destroyAllWindows()
                    break
                elif cv2.getWindowProperty("Select ROI", cv2.WND_PROP_VISIBLE) < 1:
                    # 窗口被关闭
                    if self._roi_points:
                        if is_polygon_closed(self._roi_points) and is_polygon_non_crossing(self._roi_points):
                            self.core.roi_points = self._roi_points
                            self.roi_status_label.config(text="已选取关注区域")
                            self.roi_button.config(text="取消选取", command=self.cancel_roi)
                            logging.info("ROI 选取成功 (窗口关闭)")
                        else:
                           messagebox.showwarning("警告", "选取的关注区域不合法，未应用ROI。")
                    cv2.destroyAllWindows()
                    break
        
        except Exception as e:
            show_error(f"选择关注区域时出错: {e}")
        finally:
            # 清理临时变量
            if hasattr(self, '_roi_points'):
                 del self._roi_points
            if hasattr(self, '_roi_frame'):
                 del self._roi_frame
            cv2.destroyAllWindows()  # 确保窗口关闭

    def cancel_roi(self):
        """取消ROI选择"""
        self.core.roi_points = []
        self.core.roi_mask = None
        self.roi_status_label.config(text="未选取关注区域")
        self.roi_button.config(text="选择关注区域", command=self.select_roi)
        logging.info("ROI 已取消")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 读取配置文件
    config = configparser.ConfigParser()
    config_files_read = config.read('set.ini')
    if not config_files_read:
        logging.warning("未找到或无法读取 set.ini 文件，将使用默认配置参数运行")
    
    try:
        # 从配置文件读取参数（或使用默认值）
        BATCH_SIZE = get_config_value(config, 'Parameters', 'batch_size', 2)
        FRAME_QUEUE_SIZE = get_config_value(config, 'Parameters', 'frame_queue_size', 15)
        RESULT_QUEUE_SIZE = get_config_value(config, 'Parameters', 'result_queue_size', 15)
        target_moving = get_config_value(config, 'Parameters', 'target_moving', 5)
    except Exception as e:
        logging.error(f"读取配置时发生未预期错误: {e}")
        # 使用硬编码默认值作为后备
        BATCH_SIZE, FRAME_QUEUE_SIZE, RESULT_QUEUE_SIZE, target_moving = 2, 15, 15, 5

    # --- 配置日志 ---
    logging.getLogger("ultralytics").setLevel(logging.WARNING)  # 减少ultralytics库的日志
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建并配置文件处理器（显式指定UTF-8编码）
    file_handler = logging.FileHandler('video_analysis.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 创建并配置控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    # --- 日志配置结束 ---

    # 创建主窗口
    root = ttk.Window(themename='darkly')
    root.title("COCO数据集目标检测")
    
    # 设置窗口图标
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS  # 打包后路径
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))  # 开发路径
    
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

    # 创建并运行GUI
    gui = VideoProcessorGUI(root)
    root.mainloop()