import cv2
import time
import torch

class VehicleTracker:
    def __init__(self, model):
        self.model = model
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
        self.car_count = 0
        self.truck_count = 0

        self.traffic_light_not_detected = True
        self.green_light = False
        self.red_light = False
        self.green_light_start_time = None
        self.red_light_start_time = None
        self.current_light_duration = 0
        self.previous_light_state = False

    def reset(self):
        """Reset tracking data"""
        self.first_seen.clear()
        self.counted_ids.clear()
        self.prev_bottom.clear()
        self.min_y.clear()
        self.max_y.clear()
        self.count = 0
        self.car_count = 0
        self.truck_count = 0
        self.green_light_start_time = None
        self.red_light_start_time = time.time()  # Initialize with red
        self.current_light_duration = 0
        self.previous_light_state = False

    def start_tracking(self, video_path):
        """Initialize the tracker on the video"""
        self.reset()
        return self.model.track(
            source=video_path,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            persist=self.persist_ids,
            stream=self.stream_mode,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            stream_buffer=False,
            vid_stride=2,
        )

    def detect_traffic_light(self, frame, result, line_points):
        """Detect traffic light status in the frame"""
        # Check if line points are set
        if len(line_points) < 2:
            return False

        start_x, start_y = line_points[0]
        end_x, end_y = line_points[1]

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

        """
        0: 'biker', 1: 'car', 2: 'pedestrian', 3: 'trafficLight',
        4: 'trafficLight-Green',  5: 'trafficLight-GreenLeft', 6: 'trafficLight-Red', 
        7: 'trafficLight-RedLeft', 8: 'trafficLight-Yellow', 9: 'trafficLight-YellowLeft', 
        10: 'truck', 11: 'motorcycle'
        """

        green_classes = [4, 5]
        yellow_classes = [8, 9]
        red_classes = [3, 6, 7]

        for box in result.boxes:
            class_id = int(box.cls[0])
            
            if class_id in [3, 4, 5, 6, 7, 8, 9]:  # Any traffic light
                self.traffic_light_not_detected = False

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                tl_center_x = (x1 + x2) // 2
                tl_center_y = (y1 + y2) // 2
                
                # Check if traffic light is within or close to the region of interest
                if (left_bound <= tl_center_x <= right_bound and
                    top_bound <= tl_center_y <= bottom_bound):
                          
                    if class_id in green_classes:
                        self.green_light = True
                    elif class_id in red_classes:
                        self.red_light = True
                    
                # Draw & label traffic lights in the relevant area
                traffic_color = (0, 255, 0) if self.green_light else (0, 0, 255) if self.red_light else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), traffic_color, 2)
                cv2.putText(frame, f"{self.model.names[class_id]}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, traffic_color, 2)

        # Update light duration for statistics
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

        # Return whether traffic light was found
        return not self.traffic_light_not_detected

    def process_frame(self, frame, result, line_points):
        """Process a frame to detect vehicles and count them crossing the line"""
        if len(line_points) != 2:
            return 0  # No counting if line not drawn
            
        # Draw line
        cv2.line(frame, line_points[0], line_points[1], (0, 255, 0), 2)
        line_y = int((line_points[0][1] + line_points[1][1]) // 2)
        
        frame_vehicles_count = 0 
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = box.cls[0].item()
            
            # Ignore non-vehicle classes
            if cls not in [1, 10]:  # car, truck
                continue
            
            if hasattr(box, 'id'):
                obj_id = int(box.id)
                by = y2  # Bottom y-coordinate
                cx = (x1 + x2) // 2
        
                if self.traffic_light_not_detected:
                    # If no traffic light detected, count all vehicles
                    self.green_light = True
            
                if obj_id not in self.first_seen:
                    self.first_seen.add(obj_id)
                    self.prev_bottom[obj_id] = by
                    self.min_y[obj_id] = by
                    self.max_y[obj_id] = by
                    if (line_y <= by) and (by <= line_y + self.margin_px) and self.green_light:
                        self.count += 1
                        self.counted_ids.add(obj_id)
                        frame_vehicles_count += 1

                        if cls == 1:
                            self.car_count += 1
                        elif cls == 10:
                            self.truck_count += 1

                else:
                    self.min_y[obj_id] = min(self.min_y[obj_id], by)
                    self.max_y[obj_id] = max(self.max_y[obj_id], by)
                    if obj_id not in self.counted_ids and self.min_y[obj_id] < line_y < self.max_y[obj_id] and self.green_light:
                        self.count += 1
                        self.counted_ids.add(obj_id)
                        frame_vehicles_count += 1
                        if cls == 1:
                            self.car_count += 1
                        elif cls == 10:
                            self.truck_count += 1
                    self.prev_bottom[obj_id] = by

                # Draw vehicle bounding box
                color = (0, 255, 0) if self.green_light else (0, 0, 255)  # Green if counting, red if not
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (cx, by), 3, (255, 255, 0), -1)
                
        # Show current count
        cv2.putText(frame, f"Count: {self.count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Also show traffic light status
        status_color = (0, 255, 0) if self.green_light else (0, 0, 255)
        cv2.putText(frame, f"Signal: {'GREEN' if self.green_light else 'RED'}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                    
        return frame_vehicles_count