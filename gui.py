import cv2
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from core import VideoProcessingCore
import os
import logging
from ultralytics import YOLO

yolo_classes = [
     "人","自行车","小车","摩托车","飞机","巴士","火车","货车","船","交通灯","消防栓",
    "停车标志","停车计费器","长椅","鸟","猫","狗","马","羊","牛","大象",
    "熊","斑马","长颈鹿","背包","雨伞","手提包","领带","行李箱","飞盘","滑雪板",
    "滑雪橇","运动球","风筝","棒球棒","棒球手套","滑板","冲浪板","网球拍","瓶子","酒杯",
    "茶杯","叉子","刀","勺子","碗","香蕉","苹果","三明治","橙子","西兰花",
    "胡萝卜","热狗","披萨","甜甜圈","蛋糕","椅子","沙发","盆栽植物","床","餐桌",
    "马桶","电视","笔记本电脑","鼠标","遥控器","键盘","手机","微波炉","烤箱","烤面包机",
    "水槽","冰箱","书","时钟","花瓶","剪刀","泰迪熊","吹风机","牙刷"
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='video_analysis_nowin.log')

def is_polygon_closed(roi_points):
    
    if len(roi_points) >= 3:
        return True
    return roi_points[0] == roi_points[-1]

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

class VideoProcessorGUI:
    def __init__(self, root):
        self.root = root  
        self.core = VideoProcessingCore(progress_callback=self.update_progress)  
        self.model = None  
        self.video_paths = []  
        self.current_video_index = 0  
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
                    checkbox.pack(side=ttk.LEFT, padx=5)
        self.select_button = self.create_button(self.object_frame, "全选", self.toggle_select_all, "primary")
        self.speed_group_frame = ttk.LabelFrame(self.root, text="分析倍速是使用丢帧方式实现，对快速移动目标视频慎用")
        self.speed_group_frame.pack(pady=5, fill=ttk.X,padx=5)
        self.speed_frame = ttk.Frame(self.speed_group_frame)
        self.speed_frame.pack(pady=2)
        self.speed_var = ttk.IntVar()  
        self.speed_var.set(1)  
        self.create_label(self.speed_frame, "选择分析倍速:")  
        speeds = [1, 2, 4, 8, 16, 24, 32, 48, 64]  
        for speed in speeds:
            radio = ttk.Radiobutton(self.speed_frame, text=f"{speed}X", variable=self.speed_var, value=speed)
            radio.pack(side=ttk.LEFT, padx=2)
        self.model_group_frame = ttk.LabelFrame(self.root, text="模型选择：更换模型要点击【加载模型】")
        self.model_group_frame.pack(pady=2, fill=ttk.X,padx=5)
        self.model_frame = ttk.Frame(self.model_group_frame)
        self.model_frame.pack(pady=2)
        self.model_var = ttk.StringVar()  
        self.model_var.set('yolo11n.pt')  
        models = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt', 'yolov12n.pt', 'yolov12s.pt', 'yolov12m.pt', 'yolov12l.pt', 'yolov12x.pt']  
        num_buttons_per_row = len(models) // 2
        first_row_frame = ttk.Frame(self.model_frame)
        first_row_frame.pack()
        for model in models[:num_buttons_per_row]:
            radio = ttk.Radiobutton(first_row_frame, text=model, variable=self.model_var, value=model)
            radio.pack(side=ttk.LEFT, padx=5)
        second_row_frame = ttk.Frame(self.model_frame)
        second_row_frame.pack(pady=5)
        for model in models[num_buttons_per_row:]:
            radio = ttk.Radiobutton(second_row_frame, text=model, variable=self.model_var, value=model)
            radio.pack(side=ttk.LEFT, padx=5)
        self.create_button(self.model_frame, "加载模型", self.load_selected_model, "primary")  
        self.create_label(self.model_frame, "yolo模型从n、s、m、l、x，推理速度由快变慢，检测精度由低变高。")
        self.roi_frame = ttk.Frame(self.root)
        self.roi_frame.pack(pady=2)
        self.roi_button = self.create_button(self.roi_frame, "选择关注区域", self.select_roi, "warning")
        self.roi_status_label = self.create_label(self.roi_frame, "没有选取关注区域", anchor=ttk.CENTER)
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=2)
        self.start_button = self.create_button(self.button_frame, "开始处理", self.start_processing, "primary")  
        self.pause_button = self.create_button(self.button_frame, "暂停处理", self.pause_processing, "secondary")  
        self.create_button(self.button_frame, "结束处理", self.stop_processing, "danger")  
        self.movement_frame = ttk.Frame(self.root)
        self.movement_frame.pack(pady=2)
        self.only_movement_var = ttk.BooleanVar(value=True)  
        checkbox = ttk.Checkbutton(self.movement_frame, text="只对活动目标截图", variable=self.only_movement_var)
        checkbox.pack(side=ttk.LEFT, padx=5)
        self.note_frame = ttk.Frame(self.root)
        self.note_frame.pack(pady=2)
        self.create_label(self.note_frame, "注意：分析过程中不要关闭程序，在弹出【视频分析已结束】窗口后再关闭。", anchor=ttk.CENTER)
        Progress_frame = ttk.Frame(self.root)
        Progress_frame.pack(pady=0, padx=5, fill=ttk.X)
        self.progress_label = ttk.Label(Progress_frame, text="等待处理视频")
        self.progress_label.pack(pady=5, fill=ttk.X)
        self.progress_bar = ttk.Progressbar(Progress_frame, orient=ttk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(pady=5, fill=ttk.X, expand=True)
       
    def create_frame(self, label_text):
        frame = ttk.Frame(self.root)  
        frame.pack(pady=10)  
        self.create_label(frame, label_text)  
        return frame  
    
    def create_entry(self, frame):
        entry = ttk.Entry(frame, width=50)  
        entry.pack(side=ttk.LEFT, padx=5)  
        return entry  

    def create_button(self, frame, text, command, bootstyle):
        button = ttk.Button(frame, text=text, command=command, bootstyle=bootstyle)  
        button.pack(side=ttk.LEFT, padx=5)  
        return button  
    
    def create_label(self, frame, text, anchor=None):
        label = ttk.Label(frame, text=text)  
        if anchor:  
            label.pack(anchor=anchor)  
        else:  
            label.pack(side=ttk.LEFT, padx=5)  
        return label  
    
    def select_video(self):
        try:
            self.video_paths = filedialog.askopenfilenames(filetypes=[("视频文件", "*.mp4;*.avi")])
            if self.video_paths:
                video_paths_str = ", ".join(self.video_paths)
                self.video_entry.delete(0, ttk.END)
                self.video_entry.insert(0, video_paths_str)
        except Exception as e:
            logging.error(f"选择视频文件时出错: {e}", exc_info=True)
            messagebox.showerror("错误", f"选择视频文件时出错: {e}")

    def select_save_path(self):
        try:
            save_path = filedialog.askdirectory()
            if save_path:
                self.save_path_entry.delete(0, ttk.END)
                self.save_path_entry.insert(0, save_path)
        except Exception as e:
            logging.error(f"选择保存路径时出错: {e}", exc_info=True)
            messagebox.showerror("错误", f"选择保存路径时出错: {e}")

    def get_selected_classes(self):
        return [text for text, var in self.checkbox_vars.items() if var.get()]  
    
    def load_model(self):
        try:
            model_path = os.path.join('models', 'yolo11n.pt')
            self.model = YOLO(model_path)
        except Exception as e:
            logging.error(f"加载模型时出错: {e}")
            messagebox.showerror("错误", f"加载模型时出错: {e}")

    def load_selected_model(self):
        model_name = self.model_var.get()
        try:
            model_path = os.path.join('models', model_name)
            self.model = YOLO(model_path)
            messagebox.showinfo("成功", f"模型 {model_name} 加载成功")
        except Exception as e:
            logging.error(f"加载模型 {model_name} 时出错: {e}")
            messagebox.showerror("错误", f"加载模型 {model_name} 时出错: {e}")

    def start_processing(self):
        logging.info("开始处理视频")
        save_path = self.save_path_entry.get()  
        selected_classes = self.get_selected_classes()  
        speed_multiplier = self.speed_var.get()  
        only_movement_targets = self.only_movement_var.get()  
        if not self.video_paths or not save_path:
            messagebox.showerror("错误", "请选择视频文件和截图保存路径")
            return
        if not selected_classes:
            messagebox.showerror("错误", "请选择至少一个检测对象")
            return
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)  
            except Exception as e:
                logging.error(f"创建保存路径时出错: {e}")
                messagebox.showerror("错误", f"创建保存路径时出错: {e}")
                return
        if self.model is None:
            messagebox.showerror("错误", "模型加载失败，请检查模型文件是否存在")
            return
        class_mapping = {
            0: "人",1: "自行车",2: "小车",3: "摩托车",4: "飞机",5: "巴士",6: "火车",7: "货车",8: "船",9: "交通灯",10: "消防栓",
            11: "停车标志",12: "停车计费器",13: "长椅",14: "鸟",15: "猫",16: "狗",17: "马",18: "羊",19: "牛",20: "大象",
            21: "熊",22: "斑马",23: "长颈鹿",24: "背包",25: "雨伞",26: "手提包",27: "领带",28: "行李箱",29: "飞盘",30: "滑雪板",
            31: "滑雪橇",32: "运动球",33: "风筝",34: "棒球棒",35: "棒球手套",36: "滑板",37: "冲浪板",38: "网球拍",39: "瓶子",40: "酒杯",
            41: "茶杯",42: "叉子",43: "刀",44: "勺子",45: "碗",46: "香蕉",47: "苹果",48: "三明治",49: "橙子",50: "西兰花",
            51: "胡萝卜",52: "热狗",53: "披萨",54: "甜甜圈",55: "蛋糕",56: "椅子",57: "沙发",58: "盆栽植物",59: "床",60: "餐桌",
            61: "马桶",62: "电视",63: "笔记本电脑",64: "鼠标",65: "遥控器",66: "键盘",67: "手机",68: "微波炉",69: "烤箱",70: "烤面包机",
            71: "水槽",72: "冰箱",73: "书",74: "时钟",75: "花瓶",76: "剪刀",77: "泰迪熊",78: "吹风机",79: "牙刷"
            }
        target_classes = [key for key, value in class_mapping.items() if value in selected_classes]
        self.start_button.config(state=ttk.DISABLED)
        self.roi_button.config(state=ttk.DISABLED)
        self.pause_button.config(state=NORMAL)
        self.current_video_index = 0  
        with self.core.pause_stop_lock:
            self.core.paused = False  
            self.core.stopped = False  
        self.process_next_video(save_path, target_classes, class_mapping, speed_multiplier, only_movement_targets)
        
    def process_next_video(self, save_path, target_classes, class_mapping, speed_multiplier, only_movement_targets):
        if self.current_video_index < len(self.video_paths):
            video_path = self.video_paths[self.current_video_index]
            video_filename = os.path.basename(video_path)
            self.progress_label.config(text=f"正在处理:{video_filename}")
            logging.info(f"开始处理视频: {video_path}")
            from threading import Thread
            read_thread = Thread(target=self.core.read_frames, args=(video_path, speed_multiplier))
            process_thread = Thread(target=self.core.process_frames, args=(
                self.model, target_classes, class_mapping, save_path, only_movement_targets))
            display_thread = Thread(target=self.core.display_results, args=(self.start_button, self.roi_button, self.process_next_video,
                                                                           save_path, target_classes, class_mapping,
                                                                           speed_multiplier, only_movement_targets,
                                                                           video_path))
            read_thread.start()  
            process_thread.start()
            display_thread.start()
            self.current_video_index += 1  
        else:
            logging.info("视频分析已结束")
            self.start_button.config(state=NORMAL)  
            self.roi_button.config(state=NORMAL)    
            self.progress_bar['value'] = 0  
            self.progress_label.config(text="等待处理视频") 
            messagebox.showinfo("提示", f"视频分析已结束！")  

    def pause_processing(self):
        with self.core.pause_stop_lock:
            self.core.paused = not self.core.paused  
        if self.core.paused:
            self.pause_button.config(text="继续处理")  
        else:
            self.pause_button.config(text="暂停处理")  

    def stop_processing(self):
        with self.core.pause_stop_lock:
            self.core.stopped = True
            self.core.paused = False  
            self.pause_button.config(state=ttk.DISABLED)
            self.start_button.config(state=NORMAL)
            self.roi_button.config(state=NORMAL)  
            self.current_video_index = 0  

    def select_all(self):
        for var in self.checkbox_vars.values():
            var.set(True)  
            
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
                                    messagebox.showerror("错误", "选取的关注区域不合法，请确保区域封闭且不交叉。")
                            cv2.destroyAllWindows()
                            break
            else:
                messagebox.showerror("错误", "请先选择视频文件")
        except Exception as e:
            logging.basicConfig(level=logging.ERROR)
            logging.error(f"选择关注区域时出错: {e}")
            messagebox.showerror("错误", f"选择关注区域时出错: {e}")

    def cancel_roi(self):
        self.core.roi_points = None
        self.roi_status_label.config(text="未选取关注区域")
        self.roi_button.config(text="选取关注区域", command=self.select_roi)