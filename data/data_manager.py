import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

class TrafficDataManager:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.traffic_data = []
        self.video_path = None
        self.root = None

    def set_root(self, root):
        """Set the root window for the application"""
        self.root = root
        
    def set_video_path(self, path):
        """Set the video path for the current session"""
        self.video_path = path
        # Generate new session ID when loading a new video
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def add_frame_data(self, frame_data):
        """Add frame data to the traffic data list"""
        self.traffic_data.append(frame_data)
        
    def clear_data(self):
        """Clear traffic data for new session"""
        self.traffic_data = []


    def save_report_as_image(self, fig):
        """Save the traffic report as an image file"""
        try:
            initial_dir = os.path.join(os.getcwd(), "reports")
            # print("Initial directory for saving report:", initial_dir)
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                initialdir=initial_dir,
                defaultextension=".png",
                filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("PDF Files", "*.pdf")],
                initialfile=f"traffic_report_{self.session_id}"
            )
            
            if not filename:
                return
                
            # Save figure
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Export", f"Report image saved successfully to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save report image: {e}")

    def export_to_csv(self):
        """Export traffic data to CSV file"""
        if not self.traffic_data:
            messagebox.showinfo("Export", "No data to export")
            return
            
        initial_dir = os.path.join(os.getcwd(), "reports")

        try:
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                initialdir=initial_dir,
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

    def save_session_data(self):
        """Save the session data to two CSV files"""
        if not self.traffic_data:
            return

        # compute average green/red durations
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
        avg_red = sum(red_durations) / len(red_durations) if red_durations else 0
        
        # Get total counts from last frame
        total_vehicles = self.traffic_data[-1]['total_count'] if self.traffic_data else 0

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
            total_vehicles,
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
        

        
    def show_report(self,  total_count=0, car_count=0, truck_count=0, motorcycle_count=0):
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
        total_vehicles = df['total_count'].iloc[-1]
        total_cars = df['car_count'].iloc[-1]
        total_trucks = df['truck_count'].iloc[-1]
        total_motorcycle = df['motorcycle_count'].iloc[-1]
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

        # Verifica se a última luz continuou até o final
        end_time = df.iloc[-1]['timestamp']
        if end_time > start_time:
            duration = end_time - start_time
            if current_color == 'GREEN':
                green_durations.append(duration)
            else:
                red_durations.append(duration)

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
        stats_frame = tk.Frame(report_window, bg="white", padx=20, pady=10)
        stats_frame.pack(fill=tk.X)
        
        stats_left = tk.Frame(stats_frame, bg="white")
        stats_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        stats_right = tk.Frame(stats_frame, bg="white")
        stats_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Left stats
        tk.Label(stats_left, text="Vehicle Count Summary", font=("Arial", 14, "bold"), bg="white").pack(anchor='w', pady=(0,10))
        tk.Label(stats_left, text=f"Total vehicles counted: {total_vehicles}", font=("Arial", 12), bg="white").pack(anchor='w')
        
        # Safely calculate percentages to avoid division by zero
        car_percentage = (total_cars/total_vehicles*100) if total_vehicles > 0 else 0
        truck_percentage = (total_trucks/total_vehicles*100) if total_vehicles > 0 else 0
        motorcycle_percentage = (total_motorcycle/total_vehicles*100) if total_vehicles > 0 else 0
        
        tk.Label(stats_left, text=f"Cars: {total_cars} ({car_percentage:.1f}%)", font=("Arial", 12), bg="white").pack(anchor='w')
        tk.Label(stats_left, text=f"Trucks: {total_trucks} ({truck_percentage:.1f}%)", font=("Arial", 12), bg="white").pack(anchor='w')
        tk.Label(stats_left, text=f"Motorcycles: {total_motorcycle} ({motorcycle_percentage:.1f}%)", font=("Arial", 12), bg="white").pack(anchor='w')

        avg_per_minute = (total_vehicles / total_duration) * 60 if total_duration > 0 else 0
        tk.Label(stats_left, text=f"Average vehicles per minute: {avg_per_minute:.2f}", font=("Arial", 12), bg="white").pack(anchor='w')
        
        # Right stats
        tk.Label(stats_right, text="Traffic Light Summary", font=("Arial", 14, "bold"), bg="white").pack(anchor='w', pady=(0,10))
        tk.Label(stats_right, text=f"Average green light duration: {avg_green:.2f} seconds", font=("Arial", 12), bg="white").pack(anchor='w')
        tk.Label(stats_right, text=f"Average red light duration: {avg_red:.2f} seconds", font=("Arial", 12), bg="white").pack(anchor='w')
        
        # Create charts
        charts_frame = tk.Frame(report_window, bg="white", padx=20, pady=10)
        charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create plot with 3 subplots
        fig = plt.Figure(figsize=(10, 9), tight_layout=True)
        
        # First plot: Vehicle distribution over time
        ax1 = fig.add_subplot(311)
        
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
        
        # Second plot: Vehicle class distribution over time
        ax2 = fig.add_subplot(312)
        
        if not df.empty:
            # Prepare data for stacked bar chart
            car_counts = []
            truck_counts = []
            motorcycle_counts = []
            
            for i in range(len(bins)-1):
                start, end = bins[i], bins[i+1]
                interval_df = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]
                
                # Calculate change in counts during this interval
                if not interval_df.empty:
                    # Get the change in counts during this interval
                    if i == 0:  # First interval
                        car_change = interval_df['car_count'].iloc[-1] - 0
                        truck_change = interval_df['truck_count'].iloc[-1] - 0
                        motorcycle_change = interval_df['motorcycle_count'].iloc[-1] - 0
                    else:
                        prev_end = df[df['timestamp'] < start]['timestamp'].idxmax() if not df[df['timestamp'] < start].empty else 0
                        prev_car = df.loc[prev_end, 'car_count'] if prev_end != 0 else 0
                        prev_truck = df.loc[prev_end, 'truck_count'] if prev_end != 0 else 0
                        prev_motorcycle = df.loc[prev_end, 'motorcycle_count'] if prev_end != 0 else 0
                        
                        car_change = interval_df['car_count'].iloc[-1] - prev_car if not interval_df.empty else 0
                        truck_change = interval_df['truck_count'].iloc[-1] - prev_truck if not interval_df.empty else 0
                        motorcycle_change = interval_df['motorcycle_count'].iloc[-1] - prev_motorcycle if not interval_df.empty else 0
                    
                    car_counts.append(max(0, car_change))
                    truck_counts.append(max(0, truck_change))
                    motorcycle_counts.append(max(0, motorcycle_change))
                else:
                    car_counts.append(0)
                    truck_counts.append(0)
                    motorcycle_counts.append(0)
            
            # Create stacked bar chart
            ax2.bar(bins[:-1], car_counts, width=4, alpha=0.7, color='dodgerblue', label='Cars')
            ax2.bar(bins[:-1], truck_counts, width=4, alpha=0.7, color='indianred', bottom=car_counts, label='Trucks')
            ax2.bar(bins[:-1], motorcycle_counts, width=4, alpha=0.7, color='gold', bottom=np.array(car_counts) + np.array(truck_counts), label='Motorcycles')

            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Vehicles by Class')
            ax2.set_title('Vehicle Class Distribution Over Time')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
        
        # Third plot: Traffic light influence
        ax3 = fig.add_subplot(313)
        
        # Extract periods where light is green or red
        if not df.empty:
            # Create timeline showing traffic light state and vehicle count
            # Sample at regular intervals to simplify
            max_time = df['timestamp'].max()
            times = np.arange(0, max_time, 0.5)  # Half-second intervals
            light_states = []
            counts_during_interval = []
            car_counts_interval = []
            truck_counts_interval = []
            motorcycle_counts_interval = []
            
            for t in times:
                # Find closest data point
                closest_idx = (df['timestamp'] - t).abs().idxmin()
                light_states.append(1 if df.loc[closest_idx, 'traffic_light'] == 'GREEN' else 0)
                
                # Count vehicles in this interval
                interval_start = t - 0.25
                interval_end = t + 0.25
                interval_df = df[(df['timestamp'] >= interval_start) & (df['timestamp'] < interval_end)]
                counts_during_interval.append(interval_df['vehicles_count'].sum())
                
                # For car/truck tracking, we'll use the difference in cumulative count
                before_idx = df[df['timestamp'] <= interval_start]['timestamp'].idxmax() if not df[df['timestamp'] <= interval_start].empty else None
                after_idx = df[df['timestamp'] >= interval_end]['timestamp'].idxmin() if not df[df['timestamp'] >= interval_end].empty else None
                
                car_diff = 0
                truck_diff = 0
                motorcycle_diff = 0
                
                if before_idx is not None and after_idx is not None:
                    car_diff = df.loc[after_idx, 'car_count'] - df.loc[before_idx, 'car_count']
                    truck_diff = df.loc[after_idx, 'truck_count'] - df.loc[before_idx, 'truck_count']
                    motorcycle_diff = df.loc[after_idx, 'motorcycle_count'] - df.loc[before_idx, 'motorcycle_count']
                
                car_counts_interval.append(car_diff)
                truck_counts_interval.append(truck_diff)
                motorcycle_counts_interval.append(motorcycle_diff)
            
            # Plot light state as background (red/green)
            ax3.fill_between(times, 0, 1, where=[x == 0 for x in light_states], color='red', alpha=0.3, transform=ax3.get_xaxis_transform())
            ax3.fill_between(times, 0, 1, where=[x == 1 for x in light_states], color='green', alpha=0.3, transform=ax3.get_xaxis_transform())
            
            # Plot vehicle count as line
            ax3_twin = ax3.twinx()
            ax3_twin.plot(times, counts_during_interval, color='blue', linewidth=1.5, label='All Vehicles')
            
            # Optional: Add separate lines for cars and trucks
            ax3_twin.plot(times, car_counts_interval, 'dodgerblue', linestyle='--', linewidth=1, label='Cars')
            ax3_twin.plot(times, truck_counts_interval, 'indianred', linestyle='--', linewidth=1, label='Trucks')
            ax3_twin.plot(times, motorcycle_counts_interval, 'gold', linestyle='--', linewidth=1, label='Motorcycles')
            
            ax3_twin.set_ylabel('Vehicles counted')
            ax3_twin.legend(loc='upper right')
            
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Traffic Light State')
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(['Red', 'Green'])
            ax3.set_title('Influence of Traffic Lights on Vehicle Flow')
        
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
            
            # Calculate vehicle class statistics during green vs red
            green_frames = df[df['traffic_light'] == 'GREEN']
            red_frames = df[df['traffic_light'] == 'RED']
            
            # Get first and last entries safely
            if len(green_frames) > 1:
                first_green = green_frames.iloc[0]
                last_green = green_frames.iloc[-1]
                cars_during_green = last_green['car_count'] - first_green['car_count']
                trucks_during_green = last_green['truck_count'] - first_green['truck_count']
                motorcycles_during_green = last_green['motorcycle_count'] - first_green['motorcycle_count']
            else:
                cars_during_green = 0
                trucks_during_green = 0
                motorcycles_during_green = 0
            
            if total_green_time > 0:
                tk.Label(stats_frame2, text=f"Cars per green light second: {cars_during_green/total_green_time:.2f}", font=("Arial", 12), bg="white").pack(anchor='w')
                tk.Label(stats_frame2, text=f"Trucks per green light second: {trucks_during_green/total_green_time:.2f}", font=("Arial", 12), bg="white").pack(anchor='w')
                tk.Label(stats_frame2, text=f"Motorcycles per green light second: {motorcycles_during_green/total_green_time:.2f}", font=("Arial", 12), bg="white").pack(anchor='w')

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
            
            # Add vehicle class specific recommendations
            car_percentage = total_cars / total_vehicles if total_vehicles > 0 else 0
            motorcycle_percentage = total_motorcycle / total_vehicles if total_vehicles > 0 else 0

            if car_percentage > 0.8:
                recommendations.append("High proportion of cars - consider optimizing for smaller vehicle throughput")
            elif car_percentage < 0.2:
                recommendations.append("High proportion of trucks - consider longer green phases to accommodate slower acceleration")
                
            if motorcycle_percentage > 0.5:
                recommendations.append("High proportion of motorcycles - consider optimizing traffic flow for two-wheelers")
            elif motorcycle_percentage < 0.1:
                recommendations.append("Low proportion of motorcycles - ensure traffic rules accommodate all vehicle types")

            if recommendations:
                tk.Label(stats_frame2, text="Recommendations:", font=("Arial", 12, "bold"), bg="white").pack(anchor='w', pady=(10,5))
                for i, rec in enumerate(recommendations):
                    tk.Label(stats_frame2, text=f"• {rec}", font=("Arial", 12), bg="white").pack(anchor='w')
        
        # Add button frame
        button_frame = tk.Frame(report_window, bg="white", padx=20, pady=10)
        button_frame.pack(fill=tk.X)
        
        # Add export button
        export_csv_btn = tk.Button(button_frame, text="Export Report to CSV", command=self.export_to_csv, bg="#4CAF50", fg="white")
        export_csv_btn.pack(side=tk.LEFT, padx=5, pady=15)
        
        # Add save as image button
        export_img_btn = tk.Button(button_frame, text="Save Report as Image", command=lambda: self.save_report_as_image(fig), bg="#2196F3", fg="white")
        export_img_btn.pack(side=tk.LEFT, padx=5, pady=15)
        
        return fig