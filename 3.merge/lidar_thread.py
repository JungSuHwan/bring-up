
import threading
import socket
import time
import math
import numpy as np

# --- Constants ---
STX = b'\x02'
ETX = b'\x03'

class LidarReceiver(threading.Thread):
    def __init__(self, ip='192.168.0.31', port=8000):
        super().__init__()
        self.ip = ip
        self.port = port
        self.running = False
        self.connected = False
        self.socket = None
        self.lock = threading.Lock()
        self.latest_points_3d = [] # List of [x, y, z]
        
        # Extrinsic Parameters (Position relative to Camera)
        # Unit: Meters
        self.offset_x = -0.12  # Left: Negative, Right: Positive
        self.offset_y = 0.0    # Down: Negative, Up: Positive
        self.offset_z = 0.0    # Forward: Negative, Backward: Positive (Default, can be changed)

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
                print(f"[Lidar] Connection lost: {e}")
                self.connected = False
                if self.socket:
                    self.socket.close()

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2.0)
            self.socket.connect((self.ip, self.port))
            print(f"[Lidar] Connected to {self.ip}:{self.port}")
            
            # Init Sequence
            self.send_command('SetAccessLevel,0000')
            time.sleep(0.1)
            self.send_command('SensorStart')
            self.connected = True
        except Exception as e:
            # print(f"[Lidar] Connect failed: {e}")
            pass

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()

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
                    
                    # Let's stick to this simple rotation for now.
                    # Mapping to Camera (Y-Up, Z-Forward)
                    # Lidar Angle 0 -> Usually Forward (-Z in Camera)
                    # Adjust rotation if Lidar is mounted differently
                    
                    # Standard mounting (Cable back):
                    # x = -r * sin(rad)
                    # z = -r * cos(rad)
                    
                    x_gl = (-r * math.sin(rad)) + self.offset_x
                    y_gl = self.offset_y
                    z_gl = (-r * math.cos(rad)) + self.offset_z
                    
                    points.append(x_gl) # x
                    points.append(y_gl) # y
                    points.append(z_gl) # z
                    
                angle_curr += angle_resol
                
            with self.lock:
                self.latest_points_3d = points # Flat list [x, y, z, x, y, z...] or simple list of lists
                
        except Exception as e:
            # print(f"Parse error: {e}")
            pass

    def get_latest_points(self):
        with self.lock:
            return list(self.latest_points_3d) # Return copy
