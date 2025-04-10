import ttkbootstrap as ttk
import sys
import os
from gui import VideoProcessorGUI

if __name__ == "__main__":
    root = ttk.Window(themename='darkly')  
    root.title("视频分析工具")
   
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    icon_path = os.path.join(base_path, 'icon.ico')
    try:
        root.iconbitmap(icon_path)
    except Exception as e:
        print(f"设置窗口图标时出错: {e}")

    gui = VideoProcessorGUI(root)
    root.mainloop()