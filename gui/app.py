import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import os
from threading import Thread
import time
from ultralytics import YOLO
import torch
from datetime import datetime

from gui.gui import VehicleCounterGUI
from track.tracker import VehicleTracker
from data.data_manager import TrafficDataManager

class VehicleCounterApp:
    def __init__(self, root):
        # Model initialization
        self.model = None
        self.tracker = None  # Will hold our VehicleTracker instance
        
        # video and UI-related variables
        self.video_path = None
        self.cap = None
        self.frame = None
        self.line_points = []
        self.video_loaded = False
        self.is_playing = False
        self.current_frame_index = 0
        self.total_frames = 0
        self.fps = 25
        self.trafficWarning = True

        # Initialize the data manager
        self.data_manager = TrafficDataManager()
        self.data_manager.set_root(root)
        
        # Initialize GUI (pass self as controller)
        self.gui = VehicleCounterGUI(root, self)
        
        # Load the model
        self.load_model()
        
    def load_model(self):
        try:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
            
            # Load YOLO model
            model = YOLO('models/model_main.pt').to(device)
            model.eval()
            
            # Initialize the tracker with the model
            self.model = model
            self.tracker = VehicleTracker(model)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")

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
        self.data_manager.set_video_path(path)
        self.frame = img.copy()
        self.gui.placeholder.place_forget()
        self.video_loaded = True
        self.gui.play_btn.config(state=tk.NORMAL)
        self.gui.reset_btn.config(state=tk.NORMAL)
        self.gui.timeline.config(to=self.total_frames-1)

        self.gui.video_status.config(text=f"Loaded: {os.path.basename(path)}")
        self.current_frame_index = 0
        self.gui.frame_count.config(text=f"Frame: 1/{self.total_frames}")
        self.gui.display_frame(self.frame, self.line_points)
        
        # Clear data for new video
        self.traffic_data = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def on_canvas_click(self, event):
        """Handle mouse click on canvas to draw line for counting vehicles"""
        if not self.video_loaded or self.is_playing: return
        x, y = event.x, event.y
        cw, ch = self.gui.canvas.winfo_width(), self.gui.canvas.winfo_height()
        h, w = self.frame.shape[:2]
        r = min(cw/w, ch/h)
        nw, nh = int(w*r), int(h*r)
        xo, yo = (cw-nw)//2, (ch-nh)//2
        if not (xo <= x <= xo+nw and yo <= y <= yo+nh): return
        fx, fy = int((x-xo)/r), int((y-yo)/r)
        if len(self.line_points) < 2:
            self.line_points.append((fx, fy))
            if len(self.line_points) == 2:
                self.gui.line_status.config(text="Line is drawn")
                self.gui.display_frame(self.frame, self.line_points)

    def reset_line(self):
        self.line_points.clear()
        self.gui.line_status.config(text="No line drawn")
        if self.frame is not None: 
            self.gui.display_frame(self.frame, self.line_points)
        self.data_manager.clear_data()

    def toggle_play(self):
        if not self.video_loaded: return
        self.is_playing = not self.is_playing
        self.gui.play_btn.config(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self.start_tracking()
        else:
            # Enable report buttons when finished playing
            self.gui.report_btn.config(state=tk.NORMAL)
            self.gui.export_btn.config(state=tk.NORMAL)

    def start_tracking(self):
        self.traffic_data = []  # Clear previous data
        
        # Use the tracker module to start tracking
        tracker_iterator = self.tracker.start_tracking(self.video_path)
        
        # Start the tracker in a separate thread to avoid blocking the UI
        Thread(target=self.run_tracker, args=(tracker_iterator,), daemon=True).start()

    def run_tracker(self, tracker_iterator):
        for frame_count, result in enumerate(tracker_iterator, start=1):
            if not self.is_playing: break
            frame = result.orig_img.copy()

            # Update the traffic light status using tracker
            self.tracker.detect_traffic_light(frame, result, self.line_points)
            
            # Show the light duration in UI
            self.gui.light_duration.config(text=f"Light Duration: {self.tracker.current_light_duration:.1f}s")
            
            # Update the UI traffic light display
            self.gui.root.after(0, lambda: self.gui.update_traffic_light(self.tracker.green_light))
            
            # If no traffic light detected and first time seeing this condition, show warning
            if self.tracker.traffic_light_not_detected and self.trafficWarning:
                self.trafficWarning = False
                messagebox.showinfo("Info", "No traffic light detected in the region of interest. Counting every vehicle.")

            self.current_frame_index += 1
            self.gui.timeline.set(self.current_frame_index)
            self.gui.frame_count.config(text=f"Frame: {self.current_frame_index}/{self.total_frames}")
            
            try:
                # Process the frame using tracker
                frame_vehicles_count = self.tracker.process_frame(frame, result, self.line_points)
                
                # Store frame data
                timestamp = self.current_frame_index / self.fps
                traffic_data = {
                    'frame': self.current_frame_index,
                    'timestamp': timestamp,
                    'vehicles_count': frame_vehicles_count,
                    'car_count': self.tracker.car_count,
                    'truck_count': self.tracker.truck_count,
                    'motorcycle_count': self.tracker.motorcycle_count,
                    'total_count': self.tracker.count,
                    'traffic_light': 'GREEN' if self.tracker.green_light else 'RED',
                    'light_duration': self.tracker.current_light_duration
                }

                frames_per_interval = int(self.fps * 0.5)

                if frame_count % frames_per_interval == 0:
                    self.data_manager.add_frame_data(traffic_data)
                
                # Update count in UI
                self.gui.count_status.config(text=f"Count: {self.tracker.count}")
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                pass

            self.gui.root.after(0, lambda f=frame.copy(): self.gui.display_frame(f, self.line_points))
            time.sleep(1/self.fps)
        
        self.is_playing = False
        self.gui.play_btn.config(text="Play")
        
        # Save session data
        self.data_manager.save_session_data()
        
        # Enable report buttons
        self.gui.report_btn.config(state=tk.NORMAL)
        self.gui.export_btn.config(state=tk.NORMAL)

    def on_seek(self, val):
        if self.is_playing or not self.video_loaded: return
        idx = int(val)
        self.current_frame_index = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame.copy()
            self.gui.display_frame(frame, self.line_points)
            self.gui.frame_count.config(text=f"Frame: {idx+1}/{self.total_frames}")


def main():
    root = tk.Tk()
    app = VehicleCounterApp(root)
    root.minsize(1000, 600)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"1200x700+{(sw-1200)//2}+{(sh-700)//2}")
    root.mainloop()

if __name__ == "__main__":
    main()