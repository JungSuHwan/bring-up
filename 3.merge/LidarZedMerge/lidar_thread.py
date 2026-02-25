
import threading
import socket
import time
import math
import numpy as np

# --- Constants ---
STX = b'\x02'
ETX = b'\x03'

class LidarReceiver(threading.Thread):
    def __init__(
        self,
        ip='192.168.0.31',
        port=8000,
        name='lidar',
        offset_x=-0.12,
        offset_y=0.0,
        offset_z=0.0,
        yaw_deg=0.0,
        alert_enabled=False,
        alert_min_m=0.0,
        alert_max_m=1.0,
    ):
        super().__init__()
        self.ip = ip
        self.port = port
        self.name = name
        self.running = False
        self.connected = False
        self.socket = None
        self.lock = threading.Lock()
        self.latest_points_3d = [] # List of [x, y, z]
        self.latest_alert_points_3d = [] # Points within configured range threshold
        self.last_connect_log_time = 0.0
        self.last_frame_time = None
        self.frame_rate_hz = 0.0
        
        # Extrinsic Parameters (Position relative to Camera)
        # Unit: Meters
        self.offset_x = offset_x  # Left: Negative, Right: Positive
        self.offset_y = offset_y  # Down: Negative, Up: Positive
        self.offset_z = offset_z  # Forward: Negative, Backward: Positive (Default, can be changed)
        self.yaw_deg = yaw_deg
        self.alert_enabled = bool(alert_enabled)
        self.alert_min_m = float(alert_min_m)
        self.alert_max_m = float(alert_max_m)

    def set_alert_threshold(self, enabled=None, min_m=None, max_m=None):
        with self.lock:
            if enabled is not None:
                self.alert_enabled = bool(enabled)
            if min_m is not None:
                self.alert_min_m = float(min_m)
            if max_m is not None:
                self.alert_max_m = float(max_m)

    def get_offset(self):
        with self.lock:
            return {
                "x": float(self.offset_x),
                "y": float(self.offset_y),
                "z": float(self.offset_z),
            }

    def set_offset(self, x=None, y=None, z=None):
        with self.lock:
            if x is not None:
                self.offset_x = float(x)
            if y is not None:
                self.offset_y = float(y)
            if z is not None:
                self.offset_z = float(z)

    def add_offset(self, dx=0.0, dy=0.0, dz=0.0):
        with self.lock:
            self.offset_x += float(dx)
            self.offset_y += float(dy)
            self.offset_z += float(dz)

    def get_yaw_deg(self):
        with self.lock:
            return float(self.yaw_deg)

    def set_yaw_deg(self, yaw_deg):
        with self.lock:
            self.yaw_deg = float(yaw_deg)

    def add_yaw_deg(self, dyaw_deg):
        with self.lock:
            self.yaw_deg += float(dyaw_deg)

    def run(self):
        self.running = True
        while self.running:
            if not self.connected:
                self.connect()
                time.sleep(1)
                continue

            try:
                self._receive_loop()
            except Exception as e:
                if self.running:
                    print(f"[Lidar:{self.name}] Connection lost: {e}")
                self.connected = False
                if self.socket:
                    try:
                        self.socket.close()
                    except Exception:
                        pass
                    self.socket = None

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2.0)
            self.socket.connect((self.ip, self.port))
            print(f"[Lidar:{self.name}] Connected to {self.ip}:{self.port}")
            
            # Init Sequence
            self.send_command('SetAccessLevel,0000')
            time.sleep(0.1)
            self.send_command('SensorStart')
            self.connected = True
        except Exception as e:
            now = time.time()
            if now - self.last_connect_log_time > 5.0:
                print(f"[Lidar:{self.name}] Connect failed to {self.ip}:{self.port} - {e}")
                self.last_connect_log_time = now

    def stop(self):
        self.running = False
        self.connected = False
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

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
            self.connected = False

    def _receive_loop(self):
        rx_buffer = b""
        while self.running and self.connected:
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
                        self._parse_packet(packet_bytes)
                    else:
                        break
            except socket.timeout:
                continue
            except Exception:
                if not self.running:
                    return
                raise

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
            
            # Convert to 3D Points
            points = []
            alert_points = []
            with self.lock:
                alert_enabled = bool(self.alert_enabled)
                alert_min = float(self.alert_min_m)
                alert_max = float(self.alert_max_m)
            angle_curr = angle_begin
            for r in ranges:
                if r > 0.05:
                    # Coordinate Transformation
                    # Lidar Frame:
                    #   X: Forward
                    #   Y: Left (standard planar lidar often is CCW, 0 is Front? Need verify)
                    #   Z: Up
                    
                    # Assuming standard math polar:
                    # 0 deg = X+ (Right of sensor?)
                    # Let's assume 0 is Front (Z- in camera??)
                    # Camera Frame (ZED OpenGL):
                    #   Y: Up
                    #   Z: Backward (Positive out of screen) -> So Front is -Z
                    #   X: Right
                    
                    # If Lidar 0 deg is Front, and rotates CCW (Left is +, Right is -?)
                    # Let's assume Lidar 0 is Front.
                    # x_lidar = r * cos(theta) -> Front
                    # y_lidar = r * sin(theta) -> Left
                    
                    # Mapping to Camera (Y-Up):
                    # Cam_Y = Lidar_Height (offset)
                    # Cam_Z = -x_lidar = -r * cos(theta)  (Forward is -Z)
                    # Cam_X = -y_lidar = -r * sin(theta)  (Left is -X, wait. X is Right.)
                    
                    # Let's try standard mapping first and adjust:
                    # r, theta -> x, z in horizontal plane.
                    rad = math.radians(angle_curr)
                    
                    # Standard Polar to Cartesian (Top Down View)
                    # x = r * cos(rad)
                    # y = r * sin(rad)
                    
                    # To OpenGL (Y Up):
                    # x_gl = x
                    # y_gl = self.offset_y
                    # z_gl = y 
                    
                    # However, ZED Camera: 
                    # X is Right, Y is Up, Z is Pull-back.
                    # Forward is -Z.
                    
                    # Rotation adjustment might be needed.
                    # Let's assume:
                    # Lidar 0 deg aligns with Camera Forward (-Z)
                    # Lidar 90 deg aligns with Camera Left (-X)
                    
                    # x_gl = -r * sin(rad)   (Forward 0->0, Left 90->-r)
                    # z_gl = -r * cos(rad)   (Forward 0->-r, Left 90->0)
                    
                    # Step-1 only: polar -> local Cartesian (LiDAR local frame).
                    # Extrinsic(offset/yaw) and world transform are applied in merge_viewer.py.
                    x_gl = -r * math.sin(rad)
                    y_gl = 0.0
                    z_gl = -r * math.cos(rad)
                    
                    points.append(x_gl) # x
                    points.append(y_gl) # y
                    points.append(z_gl) # z

                    if alert_enabled and (alert_min <= r <= alert_max):
                        alert_points.append(x_gl)
                        alert_points.append(y_gl)
                        alert_points.append(z_gl)
                    
                angle_curr += angle_resol
                
            with self.lock:
                self.latest_points_3d = points # Flat list [x, y, z, x, y, z...] or simple list of lists
                self.latest_alert_points_3d = alert_points
                now = time.time()
                if self.last_frame_time is not None:
                    dt = now - self.last_frame_time
                    if dt > 0:
                        inst_fps = 1.0 / dt
                        if self.frame_rate_hz <= 0.0:
                            self.frame_rate_hz = inst_fps
                        else:
                            self.frame_rate_hz = (0.85 * self.frame_rate_hz) + (0.15 * inst_fps)
                self.last_frame_time = now
                
        except Exception as e:
            # print(f"Parse error: {e}")
            pass

    def get_latest_points(self):
        with self.lock:
            return list(self.latest_points_3d) # Return copy

    def get_latest_alert_points(self):
        with self.lock:
            return list(self.latest_alert_points_3d)

    def get_status(self):
        with self.lock:
            return {
                "connected": self.connected,
                "point_count": len(self.latest_points_3d) // 3,
                "alert_point_count": len(self.latest_alert_points_3d) // 3,
                "fps": float(self.frame_rate_hz),
                "frame_time_s": float(self.last_frame_time) if self.last_frame_time is not None else None,
                "offset": {
                    "x": float(self.offset_x),
                    "y": float(self.offset_y),
                    "z": float(self.offset_z),
                },
                "yaw_deg": float(self.yaw_deg),
                "alert_threshold": {
                    "enabled": bool(self.alert_enabled),
                    "min_m": float(self.alert_min_m),
                    "max_m": float(self.alert_max_m),
                },
            }
