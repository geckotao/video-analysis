import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import logging
from threading import Lock
from queue import Queue
import queue
from tkinter import messagebox
import configparser
import concurrent.futures

def get_config_value(config, section, option, default):
    try:
        value = config.get(section, option)
        if isinstance(default, int):
            return int(value)
        elif isinstance(default, float):
            return float(value)
        return value
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        messagebox.showerror("错误", f"读取set.ini中 {section}.{option} 出错,将使用默认值 {default}")
        return default

def is_target_moving(prev_positions, object_key, current_position):
    prev_pos = prev_positions.get(object_key)
    return not (prev_pos and all(abs(curr - prev) < 5 for curr, prev in zip(current_position, prev_pos)))

def is_target_captured(captured_targets, target_identifier):
    return target_identifier not in captured_targets

def save_detection_screenshot(original_frame, save_path, video_filename, frame_number, object_type, saved_objects_count):
    file_name = os.path.join(save_path,
                             f"{os.path.splitext(video_filename)[0]}_{frame_number}_{object_type}_{saved_objects_count}.jpg")
    try:
        cv2.imwrite(file_name, original_frame)
        return True
    except Exception as e:
        logging.error(f"保存截图时出错: {e}")
        return False

config = configparser.ConfigParser()

try:
    config.read('set.ini')
    BATCH_SIZE = get_config_value(config, 'Parameters', 'batch_size', 1)
    FRAME_QUEUE_SIZE = get_config_value(config, 'Parameters', 'frame_queue_size', 5)
    RESULT_QUEUE_SIZE = get_config_value(config, 'Parameters', 'result_queue_size', 5)
except FileNotFoundError:
    messagebox.showerror("错误", "未找到set.ini文件,将使用默认配置参数运行")
    BATCH_SIZE = 1
    FRAME_QUEUE_SIZE = 5
    RESULT_QUEUE_SIZE = 5

logging.getLogger("ultralytics").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='video_analysis_nowin.log')

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

class VideoProcessingCore:
    def __init__(self, progress_callback=None):
        self.paused = False  
        self.stopped = False  
        self.frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)  
        self.result_queue = Queue(maxsize=RESULT_QUEUE_SIZE)  
        self.lock = Lock()  
        self.pause_stop_lock = Lock()  
        self.last_screenshot_time = {}  
        self.last_screenshot = {}  
        self.roi_points = [] 
        self.roi_mask = None  
        self.progress_callback = progress_callback  

    def detect_and_save(self, frames, original_frames, model, target_classes, class_mapping, current_times, save_path,
                        saved_objects, detection_info, only_movement_targets, prev_positions, captured_targets,
                        video_filename, fps):
        try:
            results = model(frames)
            for i, result in enumerate(results):
                boxes = result.boxes
                current_time = current_times[i]
                original_frame = original_frames[i]
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in target_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        object_key = f"{cls_id}_{x1}_{y1}_{x2}_{y2}"
                        object_type = class_mapping[cls_id]
                        if only_movement_targets and not is_target_moving(prev_positions, object_key, (x1, y1, x2, y2)):
                            continue
                        prev_positions[object_key] = (x1, y1, x2, y2)
                        target_identifier = f"{cls_id}_{round(x1 / 10)}_{round(y1 / 10)}_{round(x2 / 10)}_{round(y2 / 10)}"
                        if not is_target_captured(captured_targets, target_identifier):
                            continue
                        if object_key not in saved_objects:
                            info = f"{object_type} 在 {current_time:.2f} 秒时出现在位置: ({x1}, {y1}), ({x2}, {y2})"
                            detection_info.append(info)
                            frame_number = int(current_time * fps)
                            if save_detection_screenshot(original_frame, save_path, video_filename, frame_number, object_type,
                                                         len(saved_objects)):
                                saved_objects[object_key] = True
                                captured_targets.add(target_identifier)
                                self.last_screenshot_time[object_type] = current_time
                                self.last_screenshot[object_type] = original_frame.copy()
            return results
        except Exception as e:
            logging.error(f"模型推理出错: {e}")
            messagebox.showerror("错误", f"模型推理出错: {e}")
            return None

    def read_frames(self, video_path, speed_multiplier):
        logging.info(f"开始读取视频: {video_path}")
        with VideoCaptureContextManager(video_path) as cap:
            if not cap.isOpened():
                messagebox.showerror("错误", "无法打开视频文件")
                return
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_filename = os.path.basename(video_path)
            try:
                while True:
                    with self.pause_stop_lock:
                        if self.stopped:
                            break
                        if self.paused:
                            time.sleep(0.1)
                            continue
                    frames = []
                    original_frames = []
                    current_times = []
                    for _ in range(BATCH_SIZE):
                        for _ in range(int(speed_multiplier)):
                            if not cap.grab():
                                break
                        ret, frame = cap.retrieve()
                        if ret:
                            current_time = frame_count / fps
                            frames.append(frame)
                            original_frames.append(frame)
                            current_times.append(current_time)
                            frame_count += int(speed_multiplier)
                            # 计算处理进度并调用进度回调函数
                            progress = (frame_count / total_frames) * 100
                            if self.progress_callback:
                                self.progress_callback(progress)
                        else:
                            break
                    if frames:
                        try:
                            self.frame_queue.put((frames, original_frames, current_times, video_filename, fps), timeout=1)
                        except queue.Full:
                            logging.warning("帧队列已满，跳过当前帧")
                            continue
                    else:
                        break
            except Exception as e:
                 logging.error(f"读取帧时出错: {e}")
            finally:
                self.frame_queue.put(None)
                if self.stopped:
                    self.clear_roi()
        logging.info(f"视频读取结束: {video_path}")

    def process_frames(self, model, target_classes, class_mapping, save_path, only_movement_targets):
        logging.info("开始处理帧")
        saved_objects = {}  
        detection_info = []  
        prev_positions = {}  
        captured_targets = set()  
        try:
            while True:
                item = self.frame_queue.get()
                if item is None:
                    break
                frames, original_frames, current_times, video_filename, fps = item
                roi_frames = self.apply_roi(frames)
                results = self.detect_and_save(
                    roi_frames, original_frames, model, target_classes, class_mapping,
                    current_times, save_path, saved_objects, detection_info,
                    only_movement_targets, prev_positions, captured_targets, video_filename, fps
                )
                for i in range(len(original_frames)):
                    try:
                        self.result_queue.put((original_frames[i], results[i] if results else None), timeout=1)
                    except queue.Full:
                        logging.warning("结果队列已满，跳过当前结果")
                        continue
        except Exception as e:
            logging.error(f"处理帧时出错: {e}")
        finally:
            self.result_queue.put(None)
        logging.info("帧处理结束")

    def apply_roi(self, frames):
        if self.roi_points and self.roi_mask is None:
            self.create_roi_mask(frames[0].shape[:2])
        if self.roi_mask is not None:
            return [cv2.bitwise_and(frame, frame, mask=self.roi_mask) for frame in frames]
        return frames

    def create_roi_mask(self, frame_shape):
        with self.lock:
            self.roi_mask = np.zeros(frame_shape, dtype=np.uint8)
            roi_pts = np.array([self.roi_points], dtype=np.int32)
            cv2.fillPoly(self.roi_mask, roi_pts, 255)

    def display_results(self, start_button, roi_button, next_video_callback, save_path, target_classes, class_mapping,
                        speed_multiplier, only_movement_targets, video_path):
        logging.info("开始显示结果")
        try:
            while True:
                item = self.result_queue.get()
                if item is None:
                    break
                frame, results = item

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    with self.pause_stop_lock:
                        self.stopped = True
                    break
        except Exception as e:
            logging.error(f"显示结果时出错: {e}")
        finally:
            next_video_callback(save_path, target_classes, class_mapping, speed_multiplier, only_movement_targets)
            self.clear_roi()
        logging.info("结果显示结束")

    def start_processing(self, video_path, model, target_classes, class_mapping, save_path, speed_multiplier,
                         only_movement_targets, start_button, roi_button, next_video_callback):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_read = executor.submit(self.read_frames, video_path, speed_multiplier)
            future_process = executor.submit(self.process_frames, model, target_classes, class_mapping, save_path,
                                             only_movement_targets)
            future_display = executor.submit(self.display_results, start_button, roi_button, next_video_callback, save_path,
                                             target_classes, class_mapping, speed_multiplier, only_movement_targets, video_path)
            try:
                for future in concurrent.futures.as_completed([future_read, future_process, future_display]):
                    future.result()
            except Exception as e:
                logging.error(f"线程执行出错: {e}")
        self.clear_roi()

    def clear_roi(self):
        with self.lock:
            self.roi_points = []
            self.roi_mask = None

