import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
from threading import Thread
import time
from ultralytics import YOLO
import torch
import pandas as pd
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv

class VehicleCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Counter")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")

        # detection
        self.model = None
        self.tracker = None
        self.tracker_config = 'bytetrack.yaml'
        self.confidence_threshold = 0.5
        self.persist_ids = True
        self.stream_mode = True
        self.margin_px = 0

        self.first_seen = set()
        self.prev_bottom = {}
        self.min_y = {}
        self.max_y = {}
        self.counted_ids = set()
        self.count = 0

        self.traffic_light_not_detected = True
        self.trafficWarning = True

        # video and ui
        self.video_path = None
        self.cap = None
        self.frame = None
        self.line_points = []
        self.video_loaded = False
        self.is_playing = False
        self.current_frame_index = 0
        self.total_frames = 0
        self.fps = 25

        # save report data
        self.traffic_data = []  # Store frame-by-frame data
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # report
        self.green_light = False
        self.green_light_start_time = None
        self.red_light_start_time = None
        self.current_light_duration = 0
        self.previous_light_state = False
        

        self.create_ui()
        self.load_model()
        

    def load_model(self):
        try:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
            self.model = YOLO('models/yolo11l.pt').to(device)
            self.model.eval()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")

    def create_ui(self):
        # Create main window
        main = tk.Frame(self.root, bg="#f0f0f0")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Create control panel
        ctrl = tk.Frame(main, bg="#e0e0e0", width=200)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0,20))

        # Control panel elements
        tk.Label(ctrl, text="Vehicle Counter", font=("Arial",18,"bold"), bg="#e0e0e0").pack(pady=(20,30))
        tk.Button(ctrl, text="Load Video", command=self.load_video, bg="#4CAF50", fg="white", width=15).pack(pady=15)
        self.play_btn = tk.Button(ctrl, text="Play", command=self.toggle_play, bg="#2196F3", fg="white", width=15, state=tk.DISABLED)
        self.play_btn.pack(pady=15)
        self.reset_btn = tk.Button(ctrl, text="Reset Line", command=self.reset_line, bg="#FF9800", fg="white", width=15, state=tk.DISABLED)
        self.reset_btn.pack(pady=15)
        
        # Report buttons 
        self.report_btn = tk.Button(ctrl, text="Generate Report", command=self.show_report, bg="#9C27B0", fg="white", width=15, state=tk.DISABLED)
        self.report_btn.pack(pady=15)
        self.export_btn = tk.Button(ctrl, text="Export to CSV", command=self.export_to_csv, bg="#607D8B", fg="white", width=15, state=tk.DISABLED)
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
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.timeline = tk.Scale(self.video_panel, from_=0, to=0, orient=tk.HORIZONTAL, sliderlength=15, bg="#333333", fg="white", command=self.on_seek)
        self.timeline.pack(fill=tk.X, padx=10, pady=10)

        self.placeholder = tk.Label(self.canvas, text="Load a video to begin", fg="white", bg="#333333", font=("Arial",14))
        self.placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Initialize the green_light property
        self.green_light = False

    def update_traffic_light(self):
        """Update little traffic light based on green_light state"""
        if self.green_light:
            # Green light is active
            self.traffic_light.itemconfig("red_light", fill="#550000")  # Dim red
            self.traffic_light.itemconfig("green_light", fill="#00FF00")  # Bright green
        else:
            # Red light is active
            self.traffic_light.itemconfig("red_light", fill="#FF0000")  # Bright red
            self.traffic_light.itemconfig("green_light", fill="#005500")  # Dim green

    def load_video(self):
        """Load video file and initialize video capture"""

        path = filedialog.askopenfilename(filetypes=[("Videos","*.mp4 *.avi *.mov *.mkv, *.webm *.webp")])
        if not path:
            messagebox.showinfo("Info","No video selected"); return

        
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): messagebox.showerror("Error","Cannot open video"); return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        ret, img = self.cap.read()
        if not ret: messagebox.showerror("Error","Cannot read video"); return

        self.video_path = path
        self.frame = img.copy()
        self.placeholder.place_forget()
        self.video_loaded = True
        self.play_btn.config(state=tk.NORMAL)
        self.reset_btn.config(state=tk.NORMAL)
        self.timeline.config(to=self.total_frames-1)

        self.video_status.config(text=f"Loaded: {os.path.basename(path)}")
        self.current_frame_index = 0
        self.frame_count.config(text=f"Frame: 1/{self.total_frames}")
        self.display_frame(self.frame)
        
        # Clear data for new video
        self.traffic_data = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def display_frame(self, frame):
        """Display video frame on canvas"""
        h,w = frame.shape[:2]
        cw,ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        r = min(cw/w, ch/h)
        nw,nh = int(w*r), int(h*r)
        img = cv2.resize(frame, (nw,nh))
        if len(self.line_points)==2:
            p1,p2 = [(int(x*r),int(y*r)) for x,y in self.line_points]
            cv2.line(img,p1,p2,(0,255,0),2)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        tkimg = ImageTk.PhotoImage(Image.fromarray(img))
        self.photo=tkimg
        self.canvas.delete("all")
        self.canvas.create_image(cw//2,ch//2,image=tkimg)

    def on_canvas_click(self,event):
        """Handle mouse click on canvas to draw line for counting vehicles [used in tracker]"""
        if not self.video_loaded or self.is_playing: return
        x,y=event.x,event.y
        cw,ch=self.canvas.winfo_width(),self.canvas.winfo_height()
        h,w=self.frame.shape[:2]
        r=min(cw/w,ch/h)
        nw,nh=int(w*r),int(h*r)
        xo,yo=(cw-nw)//2,(ch-nh)//2
        if not (xo<=x<=xo+nw and yo<=y<=yo+nh): return
        fx,fy=int((x-xo)/r),int((y-yo)/r)
        if len(self.line_points)<2:
            self.line_points.append((fx,fy))
            if len(self.line_points)==2:
                self.line_status.config(text="Line is drawn")
                self.display_frame(self.frame)

    def reset_line(self):
        self.line_points.clear()
        self.line_status.config(text="No line drawn")
        if self.frame is not None: self.display_frame(self.frame)

    def toggle_play(self):
        if not self.video_loaded: return
        self.is_playing=not self.is_playing
        self.play_btn.config(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self.start_tracking()
        else:
            # Enable report buttons when finished playing
            self.report_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.NORMAL)

    def start_tracking(self):
        self.first_seen.clear(); self.counted_ids.clear()
        self.prev_bottom.clear(); self.min_y.clear(); self.max_y.clear()
        self.count=0; self.current_frame_index=0
        self.traffic_data = []  # Clear previous data
        self.green_light_start_time = None
        self.red_light_start_time = time.time()  # Initialize with red
        self.current_light_duration = 0
        self.previous_light_state = False

        self.tracker=self.model.track(
            source=self.video_path,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            persist=self.persist_ids,
            stream=self.stream_mode,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            stream_buffer=False,
            vid_stride=2,
        )
        Thread(target=self.run_tracker,daemon=True).start()

    def get_traffic_light_status(self, frame, result):
        """Detect traffic light status in the frame"""

        # Check if line points are set
        if len(self.line_points) < 2:
            return
            
        start_x, start_y = self.line_points[0]
        end_x, end_y = self.line_points[1]


        # Create boundaries for traffic light detection
        left_bound = min(start_x-20, end_x+20)
        right_bound = max(start_x-20, end_x+20)

        # Set vertical boundary with line as bottom and margin above (only detect traffic lights above the line)
        line_y = int((start_y + end_y) // 2)  
        top_bound = 0 
        bottom_bound = line_y 

        # Draw region of interest (optional, for debugging)
        cv2.rectangle(frame, (left_bound, top_bound), (right_bound, bottom_bound), (255, 255, 0), 1)
        
        # Store previous light state
        prev_state = self.green_light
        self.green_light = False  # Reset for this frame
        
        self.traffic_light_not_detected = True

       
        for box in result.boxes:
            class_id = int(box.cls[0])
            
            if class_id == 9:  # Traffic light class
                self.traffic_light_not_detected = False

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                tl_center_x = (x1 + x2) // 2
                tl_center_y = (y1 + y2) // 2
                
                # Check if traffic light is within or close to the region of interest
                if (left_bound <= tl_center_x <= right_bound and
                    top_bound <= tl_center_y <= bottom_bound):
                    
                    # Crop and check for green in the traffic light patch
                    patch = frame[y1:y2, x1:x2]
                    if patch.size > 0: 
                        # get HSV color space to detect green
                        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                        
                        # define hsv range to detect green
                        lower = np.array([40, 50, 50])
                        upper = np.array([85, 255, 255])
                        mask = cv2.inRange(hsv, lower, upper)
                        
                        
                        if cv2.countNonZero(mask) > (patch.size * 0.05):
                            self.green_light = True
                            break
                    
                    # Draw & label traffic lights in the relevant area
                    traffic_color = (0, 255, 0) if self.green_light else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), traffic_color, 2)
                    cv2.putText(frame, f"Traffic Light - {'GREEN' if self.green_light else 'RED'}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, traffic_color, 2)



        if self.traffic_light_not_detected and self.trafficWarning:
            self.trafficWarning = False
            messagebox.showinfo("Info", "No traffic light detected in the region of interest. Counting every vehicle.")
        
        # get traffic light duration for report
        current_time = time.time()

        # if traffic light changed
        if self.green_light != prev_state: 

            # Changed to green
            if self.green_light:  
                if self.red_light_start_time:
                    red_duration = current_time - self.red_light_start_time
                    self.current_light_duration = red_duration
                self.green_light_start_time = current_time
                self.red_light_start_time = None

            # Changed to red
            else:  
                if self.green_light_start_time:
                    green_duration = current_time - self.green_light_start_time
                    self.current_light_duration = green_duration
                self.red_light_start_time = current_time
                self.green_light_start_time = None

        # no change
        else:  
            # update duration for green or red light
            if self.green_light and self.green_light_start_time:
                self.current_light_duration = current_time - self.green_light_start_time
            elif not self.green_light and self.red_light_start_time:
                self.current_light_duration = current_time - self.red_light_start_time

        # Update the UI
        self.light_duration.config(text=f"Light Duration: {self.current_light_duration:.1f}s")
        self.root.after(0, self.update_traffic_light)

    def run_tracker(self):
        """"""
        for frame_count, result in enumerate(self.tracker, start=1):
            if not self.is_playing: break
            frame = result.orig_img.copy()

            # Check for traffic lights if self.traffic_light_not_detected is true
            # if self.traffic_light_not_detected:
            #     self.t

            # print(type(frame))

            self.green_light = False

            
            self.get_traffic_light_status(frame, result)

            self.current_frame_index += 1
            self.timeline.set(self.current_frame_index)
            self.frame_count.config(text=f"Frame: {self.current_frame_index}/{self.total_frames}")
            try:
                if len(self.line_points) == 2:
                    # draw line
                    cv2.line(frame, self.line_points[0], self.line_points[1], (0, 255, 0), 2)
                    line_y = int((self.line_points[0][1] + self.line_points[1][1]) // 2)
                    
                    frame_vehicles_count = 0 
                    
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cls = box.cls[0].item()
                        

                        # ignore non-vehicle classes
                        if cls not in [0, 2, 3, 5, 7]:
                            continue
                        

                        if hasattr(box, 'id'):
                            obj_id = int(box.id)
                            by = y2
                            cx = (x1 + x2) // 2

                            if self.traffic_light_not_detected:
                                # If no traffic light detected, count all vehicles
                                print("Traffic light not detected, counting all vehicles.")
                                self.green_light = True
                            
                            
                            if obj_id not in self.first_seen:
                                self.first_seen.add(obj_id)
                                self.prev_bottom[obj_id] = by
                                self.min_y[obj_id] = by
                                self.max_y[obj_id] = by
                                if (line_y <= by) and (by <= line_y + self.margin_px and self.green_light):
                                    self.count += 1
                                    self.counted_ids.add(obj_id)
                                    frame_vehicles_count += 1
                            else:
                                self.min_y[obj_id] = min(self.min_y[obj_id], by)
                                self.max_y[obj_id] = max(self.max_y[obj_id], by)
                                if obj_id not in self.counted_ids and self.min_y[obj_id] < line_y < self.max_y[obj_id] and self.green_light:
                                    self.count += 1
                                    self.counted_ids.add(obj_id)
                                    frame_vehicles_count += 1
                                self.prev_bottom[obj_id] = by
                            
                            print(f"Green light: {self.green_light}, Count: {self.count}, ID: {obj_id}, Bottom Y: {by}")
                            # Draw vehicle bounding box
                            color = (0, 255, 0) if self.green_light else (0, 0, 255)  # Green if counting, red if not
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"ID:{obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.circle(frame, (cx, by), 3, (255, 255, 0), -1)
                    
                    # Store frame data
                    timestamp = self.current_frame_index / self.fps
                    self.traffic_data.append({
                        'frame': self.current_frame_index,
                        'timestamp': timestamp,
                        'vehicles_count': frame_vehicles_count,
                        'total_count': self.count,
                        'traffic_light': 'GREEN' if self.green_light else 'RED',
                        'light_duration': self.current_light_duration
                    })
                    
                    # Show current count
                    cv2.putText(frame, f"Count: {self.count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Also show traffic light status
                    status_color = (0, 255, 0) if self.green_light else (0, 0, 255)
                    cv2.putText(frame, f"Signal: {'GREEN' if self.green_light else 'RED'}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                    
                    self.count_status.config(text=f"Count: {self.count}")
            except:
                pass

            self.root.after(0, lambda f=frame.copy(): self.display_frame(f))
            time.sleep(1/self.fps)
        
        self.is_playing = False
        self.play_btn.config(text="Play")
        
        # Save session data to database
        self.save_session_data()
        
        # Enable report buttons
        self.report_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)

    def save_session_data(self):
        """Save the session data to two CSV files instead of the database"""
        if not self.traffic_data:
            return

        # compute average green/red durations exactly as you did
        green_durations, red_durations = [], []
        current_color = None
        start_time = 0

        for i, data in enumerate(self.traffic_data):
            if i == 0:
                current_color = data['traffic_light']
                start_time = data['timestamp']
                continue

            if data['traffic_light'] != current_color:
                duration = data['timestamp'] - start_time
                if current_color == 'GREEN':
                    green_durations.append(duration)
                else:
                    red_durations.append(duration)
                current_color = data['traffic_light']
                start_time = data['timestamp']

        avg_green = sum(green_durations) / len(green_durations) if green_durations else 0
        avg_red   = sum(red_durations)   / len(red_durations)   if red_durations   else 0

        # --- 1) Write (or append) the session summary ---
        sessions_file = 'traffic_analysis.csv'
        session_headers = [
            'session_id','video_path','date_time','total_vehicles',
            'avg_green_duration','avg_red_duration'
        ]
        session_row = [
            self.session_id,
            self.video_path,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.count,
            avg_green,
            avg_red
        ]
        # open in append mode, write header only if file is new
        write_header = not os.path.exists(sessions_file)
        with open(sessions_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(session_headers)
            writer.writerow(session_row)

        # --- 2) Write (or append) all frame-level data ---
        frames_file = 'frame_data.csv'
        frame_headers = [
            'session_id','frame_number','timestamp',
            'vehicles_count','traffic_light','light_duration'
        ]
        write_header = not os.path.exists(frames_file)
        with open(frames_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(frame_headers)
            for data in self.traffic_data:
                writer.writerow([
                    self.session_id,
                    data['frame'],
                    data['timestamp'],
                    data['vehicles_count'],
                    data['traffic_light'],
                    data['light_duration']
                ])

        print(f"Session data appended to {sessions_file} and {frames_file}")

    def export_to_csv(self):
        """Export traffic data to CSV file"""
        if not self.traffic_data:
            messagebox.showinfo("Export", "No data to export")
            return
            
        try:
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv")],
                initialfile=f"traffic_data_{self.session_id}.csv"
            )
            
            if not filename:
                return
                
            # Create DataFrame and save to CSV
            df = pd.DataFrame(self.traffic_data)
            df.to_csv(filename, index=False)
            messagebox.showinfo("Export", f"Data exported successfully to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {e}")

    def show_report(self):
        """Show traffic analysis report"""
        if not self.traffic_data:
            messagebox.showinfo("Report", "No data to generate report")
            return
            
        # Create report window
        report_window = tk.Toplevel(self.root)
        report_window.title("Traffic Analysis Report")
        report_window.geometry("900x700")
        report_window.configure(bg="white")
        
        # Create DataFrame
        df = pd.DataFrame(self.traffic_data)
        
        # Basic stats
        total_vehicles = self.count
        video_name = os.path.basename(self.video_path) if self.video_path else "Unknown"
        total_duration = df['timestamp'].max() if not df.empty else 0
        
        # Calculate light cycle times
        green_durations = []
        red_durations = []
        current_color = None
        start_time = 0
        
        for i, row in df.iterrows():
            if i == 0:
                current_color = row['traffic_light']
                start_time = row['timestamp']
                continue
                
            if row['traffic_light'] != current_color:  # Light changed
                duration = row['timestamp'] - start_time
                if current_color == 'GREEN':
                    green_durations.append(duration)
                else:
                    red_durations.append(duration)
                current_color = row['traffic_light']
                start_time = row['timestamp']
        
        avg_green = sum(green_durations) / len(green_durations) if green_durations else 0
        avg_red = sum(red_durations) / len(red_durations) if red_durations else 0
        
        # Create report header
        header_frame = tk.Frame(report_window, bg="white", padx=20, pady=20)
        header_frame.pack(fill=tk.X)
        
        tk.Label(header_frame, text="Traffic Analysis Report", font=("Arial", 18, "bold"), bg="white").pack(anchor='w')
        tk.Label(header_frame, text=f"Video: {video_name}", font=("Arial", 12), bg="white").pack(anchor='w', pady=(10,0))
        tk.Label(header_frame, text=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", font=("Arial", 12), bg="white").pack(anchor='w')
        tk.Label(header_frame, text=f"Duration: {total_duration:.1f} seconds", font=("Arial", 12), bg="white").pack(anchor='w')
        
        # Create summary stats
        # Create summary stats
        stats_frame = tk.Frame(report_window, bg="white", padx=20, pady=10)
        stats_frame.pack(fill=tk.X)
        
        stats_left = tk.Frame(stats_frame, bg="white")
        stats_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        stats_right = tk.Frame(stats_frame, bg="white")
        stats_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Left stats
        tk.Label(stats_left, text="Vehicle Count Summary", font=("Arial", 14, "bold"), bg="white").pack(anchor='w', pady=(0,10))
        tk.Label(stats_left, text=f"Total vehicles counted: {total_vehicles}", font=("Arial", 12), bg="white").pack(anchor='w')
        avg_per_minute = (total_vehicles / total_duration) * 60 if total_duration > 0 else 0
        tk.Label(stats_left, text=f"Average vehicles per minute: {avg_per_minute:.2f}", font=("Arial", 12), bg="white").pack(anchor='w')
        
        # Right stats
        tk.Label(stats_right, text="Traffic Light Summary", font=("Arial", 14, "bold"), bg="white").pack(anchor='w', pady=(0,10))
        tk.Label(stats_right, text=f"Average green light duration: {avg_green:.2f} seconds", font=("Arial", 12), bg="white").pack(anchor='w')
        tk.Label(stats_right, text=f"Average red light duration: {avg_red:.2f} seconds", font=("Arial", 12), bg="white").pack(anchor='w')
        
        # Create charts
        charts_frame = tk.Frame(report_window, bg="white", padx=20, pady=10)
        charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Plot 1: Vehicle count over time
        fig = plt.Figure(figsize=(10, 6), tight_layout=True)
        
        # First plot: Vehicle distribution over time
        ax1 = fig.add_subplot(211)
        
        # Calculate vehicles per 5-second window
        if not df.empty:
            max_time = df['timestamp'].max()
            bins = np.arange(0, max_time + 5, 5)  # 5-second windows
            hist_data = []
            for i in range(len(bins)-1):
                start, end = bins[i], bins[i+1]
                count = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]['vehicles_count'].sum()
                hist_data.append(count)
            
            ax1.bar(bins[:-1], hist_data, width=4, alpha=0.7, color='cornflowerblue')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Vehicles Counted')
            ax1.set_title('Vehicle Distribution Over Time')
            ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Second plot: Traffic light influence
        ax2 = fig.add_subplot(212)
        
        # Extract periods where light is green or red
        if not df.empty:
            # Create timeline showing traffic light state and vehicle count
            # Sample at regular intervals to simplify
            times = np.arange(0, max_time, 0.5)  # Half-second intervals
            light_states = []
            counts_during_interval = []
            
            for t in times:
                # Find closest data point
                closest_idx = (df['timestamp'] - t).abs().idxmin()
                light_states.append(1 if df.loc[closest_idx, 'traffic_light'] == 'GREEN' else 0)
                
                # Count vehicles in this interval
                interval_start = t - 0.25
                interval_end = t + 0.25
                interval_df = df[(df['timestamp'] >= interval_start) & (df['timestamp'] < interval_end)]
                counts_during_interval.append(interval_df['vehicles_count'].sum())
            
            # Plot light state as background (red/green)
            ax2.fill_between(times, 0, 1, where=[x == 0 for x in light_states], color='red', alpha=0.3, transform=ax2.get_xaxis_transform())
            ax2.fill_between(times, 0, 1, where=[x == 1 for x in light_states], color='green', alpha=0.3, transform=ax2.get_xaxis_transform())
            
            # Plot vehicle count as line
            ax2_twin = ax2.twinx()
            ax2_twin.plot(times, counts_during_interval, color='blue', linewidth=1.5)
            ax2_twin.set_ylabel('Vehicles counted')
            
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Traffic Light State')
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Red', 'Green'])
            ax2.set_title('Influence of Traffic Lights on Vehicle Flow')
        
        canvas = FigureCanvasTkAgg(fig, master=charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Calculate traffic light efficiency
        if total_vehicles > 0 and avg_green > 0:
            # Traffic flow efficiency statistics
            total_green_time = sum(green_durations) if green_durations else 0
            vehicles_per_green_second = total_vehicles / total_green_time if total_green_time > 0 else 0
            
            # Calculate optimal light timing (this is a simple approximation)
            light_ratio = avg_green / (avg_green + avg_red) if (avg_green + avg_red) > 0 else 0
            
            stats_frame2 = tk.Frame(report_window, bg="white", padx=20, pady=10)
            stats_frame2.pack(fill=tk.X)
            
            tk.Label(stats_frame2, text="Traffic Flow Efficiency", font=("Arial", 14, "bold"), bg="white").pack(anchor='w', pady=(0,10))
            tk.Label(stats_frame2, text=f"Vehicles per green light second: {vehicles_per_green_second:.2f}", font=("Arial", 12), bg="white").pack(anchor='w')
            tk.Label(stats_frame2, text=f"Green/red light ratio: {light_ratio:.2f}", font=("Arial", 12), bg="white").pack(anchor='w')
            
            # Recommendations based on data
            recommendations = []
            if vehicles_per_green_second < 0.1:
                recommendations.append("Green light duration may be too long for current traffic volume")
            elif vehicles_per_green_second > 0.5:
                recommendations.append("Green light duration may be too short for current traffic volume")
                
            if light_ratio < 0.3:
                recommendations.append("Consider increasing green light duration relative to red light")
            elif light_ratio > 0.7:
                recommendations.append("Consider more balanced green/red light timing")
                
            if recommendations:
                tk.Label(stats_frame2, text="Recommendations:", font=("Arial", 12, "bold"), bg="white").pack(anchor='w', pady=(10,5))
                for i, rec in enumerate(recommendations):
                    tk.Label(stats_frame2, text=f"• {rec}", font=("Arial", 12), bg="white").pack(anchor='w')
        
        # Add export button
        export_btn = tk.Button(report_window, text="Export Report to CSV", command=self.export_to_csv, bg="#4CAF50", fg="white")
        export_btn.pack(pady=15)

    def on_seek(self, val):
        if self.is_playing or not self.video_loaded: return
        idx = int(val)
        self.current_frame_index = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame.copy()
            self.display_frame(frame)
            self.frame_count.config(text=f"Frame: {idx+1}/{self.total_frames}")

    def calculate_flow_statistics(self):
        """Calculate traffic flow statistics"""
        if not self.traffic_data:
            return None
            
        df = pd.DataFrame(self.traffic_data)
        
        # Group data by traffic light state
        green_frames = df[df['traffic_light'] == 'GREEN']
        red_frames = df[df['traffic_light'] == 'RED']
        
        # Calculate vehicles per minute during green and red
        green_duration = green_frames['timestamp'].max() - green_frames['timestamp'].min() if len(green_frames) > 1 else 0
        total_green_vehicles = green_frames['vehicles_count'].sum()
        
        green_vehicles_per_minute = (total_green_vehicles / green_duration) * 60 if green_duration > 0 else 0
        
        # Calculate light cycle times
        light_changes = df['traffic_light'] != df['traffic_light'].shift(1)
        light_change_idx = df[light_changes].index.tolist()
        
        cycles = []
        for i in range(len(light_change_idx) - 1):
            start_idx = light_change_idx[i]
            end_idx = light_change_idx[i+1]
            
            light_state = df.loc[start_idx, 'traffic_light']
            duration = df.loc[end_idx, 'timestamp'] - df.loc[start_idx, 'timestamp']
            vehicles = df.loc[start_idx:end_idx, 'vehicles_count'].sum()
            
            cycles.append({
                'state': light_state,
                'duration': duration,
                'vehicles': vehicles
            })
            
        stats = {
            'green_vehicles_per_minute': green_vehicles_per_minute,
            'cycles': cycles
        }
        
        return stats

def main():
    root = tk.Tk()
    app = VehicleCounterApp(root)
    root.minsize(1000, 600)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"1200x700+{(sw-1200)//2}+{(sh-700)//2}")
    root.mainloop()

if __name__ == "__main__":
    main()