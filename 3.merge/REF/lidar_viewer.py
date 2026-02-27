import sys
import socket
import time
import math
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import messagebox

# --- Constants ---
STX = b'\x02'
ETX = b'\x03'
CANVAS_SIZE = 800
MAX_RANGE_M = 10.0  # Max display range
COLORS = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)] # BGR Colors

class LidarSensor:
    def __init__(self, ip, port=8000, sensor_id=0):
        self.ip = ip
        self.port = port
        self.id = sensor_id
        self.socket = None
        self.connected = False
        self.running = False
        self.latest_scan = None
        self.lock = threading.Lock()
        self.recv_thread = None

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2.0)
            self.socket.connect((self.ip, self.port))
            print(f"[{self.ip}] Connected.")
            
            # Init Sequence
            self.send_command('SetAccessLevel,0000')
            time.sleep(0.1)
            self.send_command('SensorStart')
            
            self.connected = True
            self.running = True
            self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.recv_thread.start()
            return True
        except Exception as e:
            print(f"[{self.ip}] Connection Failed: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False

    def send_command(self, cmd):
        if not self.socket: return
        try:
            cmd_type = 'sMC'
            if 'SensorScanInfo' in cmd:
                cmd_type = 'sRC'
            elif cmd.startswith('LSScanDataConfig'):
                cmd_type = 'sWC' if ',' in cmd else 'sRC'
                
            payload = f",{cmd_type},{cmd}"
            total_len = 1 + 4 + len(payload) + 1
            len_str = f"{total_len:04X}"
            packet = STX + len_str.encode('ascii') + payload.encode('ascii') + ETX
            self.socket.send(packet)
        except:
            self.disconnect()

    def _receive_loop(self):
        rx_buffer = b""
        while self.running:
            try:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                rx_buffer += chunk
                
                while True:
                    stx_idx = rx_buffer.find(STX)
                    if stx_idx == -1:
                        if len(rx_buffer) > 8192: rx_buffer = b""
                        break
                    
                    etx_idx = rx_buffer.find(ETX, stx_idx)
                    if etx_idx != -1:
                        packet_bytes = rx_buffer[stx_idx+1 : etx_idx]
                        rx_buffer = rx_buffer[etx_idx+1:]
                        try:
                            self._parse_packet(packet_bytes)
                        except:
                            pass
                    else:
                        break
            except socket.timeout:
                continue
            except Exception:
                break
        self.connected = False

    def _parse_packet(self, packet_bytes):
        try:
            packet_str = packet_bytes.decode('ascii')
            if 'DIST1' not in packet_str: return

            fields = packet_str.split(',')
            dist_index = fields.index('DIST1')
            
            # Header Parsing
            angle_begin_raw = int(fields[dist_index - 4], 16)
            if angle_begin_raw > 0x7FFFFFFF:
                angle_begin_raw -= 0xFFFFFFFF + 1
            angle_begin = angle_begin_raw / 10000.0
            
            angle_resol = int(fields[dist_index - 3], 16) / 10000.0
            amount = int(fields[dist_index - 2], 16)
            
            ranges = []
            start_idx = dist_index + 1
            for i in range(amount):
                if start_idx + i >= len(fields): break
                val = fields[start_idx + i]
                if not val: continue
                dist_m = int(val, 16) / 1000.0
                ranges.append(dist_m)
                
            with self.lock:
                self.latest_scan = {
                    'angle_begin': angle_begin,
                    'angle_resol': angle_resol,
                    'ranges': ranges
                }
        except:
            pass

    def get_scan(self):
        with self.lock:
            return self.latest_scan

# --- Viewer Logic ---
def run_viewer_loop(ip_list):
    sensors = []
    for i, ip in enumerate(ip_list):
        ip = ip.strip()
        if not ip: continue
        s = LidarSensor(ip, sensor_id=i)
        if s.connect():
            sensors.append(s)
        else:
            print(f"Failed to connect to {ip}")
            
    if not sensors:
        messagebox.showerror("Error", "No sensors connected.")
        return

    # Window Setup
    win_name = "Multi-Lidar View"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    # Mouse State
    mouse_x, mouse_y = 0, 0
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y
        mouse_x, mouse_y = x, y

    cv2.setMouseCallback(win_name, mouse_callback)

    # Main Loop
    img = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    center_x, center_y = CANVAS_SIZE // 2, CANVAS_SIZE // 2
    pixels_per_meter = (CANVAS_SIZE / 2) / MAX_RANGE_M
    
    frame_cnt = 0
    fps = 0.0
    last_fps_time = time.time()
    
    running = True
    while running:
        start_t = time.time()
        
        # 1. Background
        img.fill(30) # Dark Gray
        
        # 2. Grid & Crosshair
        for r in range(1, int(MAX_RANGE_M) + 1):
            rad = int(r * pixels_per_meter)
            cv2.circle(img, (center_x, center_y), rad, (60, 60, 60), 1)
            cv2.putText(img, f"{r}m", (center_x + 5, center_y - rad), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
        cv2.line(img, (center_x, 0), (center_x, CANVAS_SIZE), (50, 50, 50), 1)
        cv2.line(img, (0, center_y), (CANVAS_SIZE, center_y), (50, 50, 50), 1)

        # 3. Draw Sensor Data
        for s in sensors:
            scan = s.get_scan()
            if scan:
                color = COLORS[s.id % len(COLORS)]
                angle = scan['angle_begin']
                step = scan['angle_resol']
                
                # Pre-calculate trig could be faster but basic math is fine for standard lidar resolution
                angles = np.arange(len(scan['ranges'])) * step + angle
                ranges = np.array(scan['ranges'])
                
                # Filter valid
                valid = (ranges > 0.05) & (ranges < MAX_RANGE_M * 1.5)
                r_valid = ranges[valid]
                a_valid = np.radians(angles[valid])
                
                # Polar to Cartesian
                # Math: X=cos, Y=sin. 
                # Screen: X=cx+x, Y=cy-y
                x = r_valid * np.cos(a_valid)
                y = r_valid * np.sin(a_valid)
                
                px = (center_x + x * pixels_per_meter).astype(np.int32)
                py = (center_y - y * pixels_per_meter).astype(np.int32)
                
                # Bulk draw using line iterator or simply loop
                # For cleaner points in Python cv2, circle loop is okay or modify pixels directly
                for ix, iy in zip(px, py):
                    cv2.circle(img, (ix, iy), 2, color, -1)

        # 4. Mouse Info Overlay
        # Distance from center
        dx = mouse_x - center_x
        dy = center_y - mouse_y # Up is Positive Y
        dist_m = math.sqrt(dx*dx + dy*dy) / pixels_per_meter
        angle_deg = math.degrees(math.atan2(dy, dx))
        if angle_deg < 0: angle_deg += 360
        
        # Draw Line to Mouse
        cv2.line(img, (center_x, center_y), (mouse_x, mouse_y), (100, 100, 100), 1)
        
        # Info Box
        info_text = f"{dist_m:.2f}m / {angle_deg:.1f} deg"
        cv2.putText(img, info_text, (mouse_x + 10, mouse_y - 10), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 5. FPS & Sensor Status
        frame_cnt += 1
        curr_t = time.time()
        if curr_t - last_fps_time >= 1.0:
            fps = frame_cnt / (curr_t - last_fps_time)
            frame_cnt = 0
            last_fps_time = curr_t
            
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        y_offset = 60
        for s in sensors:
            status = "OK" if s.connected else "LOST"
            col = COLORS[s.id % len(COLORS)]
            cv2.putText(img, f"ID {s.id} [{s.ip}]: {status}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            y_offset += 25

        # 6. Show
        cv2.imshow(win_name, img)
        key = cv2.waitKey(10)
        if key == 27 or key == ord('q'):
            running = False
            
    # Cleanup
    for s in sensors:
        s.disconnect()
    cv2.destroyAllWindows()


# --- Launcher UI ---
def main():
    root = tk.Tk()
    root.title("Lidar Launcher")
    root.geometry("300x150")
    
    lbl = tk.Label(root, text="Enter Lidar IPs (comma separated):")
    lbl.pack(pady=10)
    
    entry = tk.Entry(root, width=40)
    entry.insert(0, "192.168.0.31") 
    entry.pack(pady=5)
    
    def on_start():
        ip_str = entry.get()
        if not ip_str: return
        ip_list = ip_str.split(',')
        root.destroy()
        run_viewer_loop(ip_list)
        
    btn = tk.Button(root, text="Connect & Start", command=on_start, height=2)
    btn.pack(pady=20, fill='x', padx=20)
    
    root.mainloop()

if __name__ == "__main__":
    main()
