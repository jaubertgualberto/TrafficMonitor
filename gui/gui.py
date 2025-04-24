import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
from threading import Thread
import time
import torch
from datetime import datetime

class VehicleCounterGUI:
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller  # Reference to the main controller
        self.root.title("Vehicle Counter")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        # UI elements that need to be accessed by controller
        self.video_status = None
        self.line_status = None
        self.count_status = None
        self.frame_count = None
        self.light_duration = None
        self.traffic_light = None
        self.play_btn = None
        self.report_btn = None
        self.export_btn = None
        self.reset_btn = None
        self.timeline = None
        self.canvas = None
        self.placeholder = None
        self.photo = None
        
        self.create_ui()
        
    def create_ui(self):
        # Create main window
        main = tk.Frame(self.root, bg="#f0f0f0")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Create control panel
        ctrl = tk.Frame(main, bg="#e0e0e0", width=200)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0,20))

        # Control panel elements
        tk.Label(ctrl, text="Vehicle Counter", font=("Arial",18,"bold"), bg="#e0e0e0").pack(pady=(20,30))
        tk.Button(ctrl, text="Load Video", command=self.controller.load_video, bg="#4CAF50", fg="white", width=15).pack(pady=15)
        self.play_btn = tk.Button(ctrl, text="Play", command=self.controller.toggle_play, bg="#2196F3", fg="white", width=15, state=tk.DISABLED)
        self.play_btn.pack(pady=15)
        self.reset_btn = tk.Button(ctrl, text="Reset Line", command=self.controller.reset_line, bg="#FF9800", fg="white", width=15, state=tk.DISABLED)
        self.reset_btn.pack(pady=15)
        
        # Report buttons 
        self.report_btn = tk.Button(ctrl, text="Generate Report", command=self.controller.data_manager.show_report, bg="#9C27B0", fg="white", width=15, state=tk.DISABLED)
        self.report_btn.pack(pady=15)
        self.export_btn = tk.Button(ctrl, text="Export to CSV", command=self.controller.data_manager.export_to_csv, bg="#607D8B", fg="white", width=15, state=tk.DISABLED)
        self.export_btn.pack(pady=15)
        
        # Status panel
        st = tk.LabelFrame(ctrl, text="Status", bg="#e0e0e0")
        st.pack(fill=tk.X, pady=(20,0), padx=10)
        self.video_status = tk.Label(st, text="No video loaded", bg="#e0e0e0")
        self.video_status.pack(anchor='w', pady=5)

        self.line_status = tk.Label(st, text="No line drawn", bg="#e0e0e0")
        self.line_status.pack(anchor='w', pady=5)
        
        self.count_status = tk.Label(st, text="Count: 0", bg="#e0e0e0")
        self.count_status.pack(anchor='w', pady=5)

        # Add little traffic light to the status panel
        tl_frame = tk.Frame(st, bg="#e0e0e0")
        tl_frame.pack(anchor='w', pady=5, fill=tk.X)
        
        # Create traffic light housing
        self.traffic_light = tk.Canvas(tl_frame, width=45, height=90, bg="#e0e0e0", highlightthickness=0)
        self.traffic_light.pack(side=tk.BOTTOM, padx=40)
        self.traffic_light.create_rectangle(0, 0, 45, 90, fill="black", outline="gray", width=2)
        self.traffic_light.create_oval(8, 8, 38, 38, fill="#550000", tags="red_light")
        self.traffic_light.create_oval(8, 53, 38, 83, fill="#005500", tags="green_light")
        
        # Add traffic light status label
        self.light_duration = tk.Label(st, text="Light Duration: 0s", bg="#e0e0e0")
        self.light_duration.pack(anchor='w', pady=5)
        self.frame_count = tk.Label(ctrl, text="Frame: 0/0", bg="#e0e0e0")
        self.frame_count.pack(pady=(20,0))

        inst = tk.LabelFrame(ctrl, text="Instructions", bg="#e0e0e0")
        inst.pack(fill=tk.X, pady=20, padx=10, expand=True)
        tk.Label(inst, text="1. Load video\n2. Draw line\n3. Play to count\n4. Generate report", justify=tk.LEFT, bg="#e0e0e0").pack()

        self.video_panel = tk.Frame(main, bg="#333333")
        self.video_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.video_panel, bg="#333333", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.controller.on_canvas_click)

        self.timeline = tk.Scale(self.video_panel, from_=0, to=0, orient=tk.HORIZONTAL, sliderlength=15, bg="#333333", fg="white", command=self.controller.on_seek)
        self.timeline.pack(fill=tk.X, padx=10, pady=10)

        self.placeholder = tk.Label(self.canvas, text="Load a video to begin", fg="white", bg="#333333", font=("Arial",14))
        self.placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
    def display_frame(self, frame, line_points=None):
        """Display video frame on canvas"""
        h, w = frame.shape[:2]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        r = min(cw/w, ch/h)
        nw, nh = int(w*r), int(h*r)
        img = cv2.resize(frame, (nw, nh))
        if line_points and len(line_points) == 2:
            p1, p2 = [(int(x*r), int(y*r)) for x, y in line_points]
            cv2.line(img, p1, p2, (0, 255, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tkimg = ImageTk.PhotoImage(Image.fromarray(img))
        self.photo = tkimg
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=tkimg)
        
    def update_traffic_light(self, is_green):
        """Update little traffic light based on tracker's green_light state"""
        if is_green:
            # Green light is active
            self.traffic_light.itemconfig("red_light", fill="#550000")  # Dim red
            self.traffic_light.itemconfig("green_light", fill="#00FF00")  # Bright green
        else:
            # Red light is active
            self.traffic_light.itemconfig("red_light", fill="#FF0000")  # Bright red
            self.traffic_light.itemconfig("green_light", fill="#005500")  # Dim green