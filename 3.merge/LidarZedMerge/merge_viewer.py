"""
[1단계: 녹화 로깅 기능]**과 [2단계: 오프라인 분석 스크립트] 구현 준비
"""
import sys
import time
import json
import os
import math
from collections import deque
import numpy as np
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import lidar_thread
import web_stream

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    print("\n[Error] 'scipy' 라이브러리가 설치되어 있지 않습니다!")
    print("LiDAR 캘리브레이션(KD-Tree 연산)을 위해 필수입니다.")
    print("설치 명령어: pip install scipy\n")
    sys.exit(1)

if os.name == "nt":
    import msvcrt
else:
    import select
    import termios
    import tty

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lidar_config.json")

class DataLogger:
    def __init__(self, log_root="logs"):
        self.log_root = log_root
        self.is_logging = False
        self.session_dir = None
        self.lidar_dir = None
        self.timestamps = []
        self.poses = []

    def start_logging(self):
        if self.is_logging: return
        self.is_logging = True
        
        time_str = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.log_root, f"session_{time_str}")
        self.lidar_dir = os.path.join(self.session_dir, "lidar_points")
        os.makedirs(self.lidar_dir, exist_ok=True)
        
        self.timestamps.clear()
        self.poses.clear()
        print(f"🔴 [DataLogger] Record started: {self.session_dir}")

    def log_frame(self, timestamp, t_world_robot_4x4, lidar_pts_nx3):
        if not self.is_logging: return
        
        self.timestamps.append(timestamp)
        self.poses.append(np.array(t_world_robot_4x4, dtype=np.float32))
        
        if len(lidar_pts_nx3) > 0:
            np.save(os.path.join(self.lidar_dir, f"{timestamp:.6f}.npy"), np.array(lidar_pts_nx3, dtype=np.float32))

    def stop_logging(self, zed_map_points=None):
        if not self.is_logging: return
        self.is_logging = False
        
        np.savez_compressed(
            os.path.join(self.session_dir, "trajectory.npz"),
            timestamps=np.array(self.timestamps, dtype=np.float64),
            poses=np.array(self.poses, dtype=np.float32)
        )
        
        if zed_map_points is not None and len(zed_map_points) > 0:
            np.save(os.path.join(self.session_dir, "zed_map.npy"), np.array(zed_map_points, dtype=np.float32))
            
        print(f"⬛ [DataLogger] Record stopped: {len(self.timestamps)} frames saved.")



def _to_np_4x4(mat_like):
    arr = np.array(mat_like, dtype=np.float32)
    if arr.shape == (4, 4):
        return arr
    flat = arr.reshape(-1)
    if flat.size != 16:
        raise ValueError(f"Expected 16 elements for 4x4 matrix, got {flat.size}")
    return flat.reshape(4, 4)


def build_extrinsic_4x4(offset, yaw_deg):
    yaw_rad = math.radians(float(yaw_deg))
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    t = np.eye(4, dtype=np.float32)
    # Y-up rotation (X-Z plane)
    t[0, 0] = c
    t[0, 2] = -s
    t[2, 0] = s
    t[2, 2] = c
    t[0, 3] = float(offset.get("x", 0.0))
    t[1, 3] = float(offset.get("y", 0.0))
    t[2, 3] = float(offset.get("z", 0.0))
    return t


def transform_points_flat(points, t_4x4):
    if not points:
        return []
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    pts_w = (t_4x4 @ pts_h.T).T[:, :3]
    return pts_w.reshape(-1).tolist()


def load_robot_overlay_options(config):
    overlay_cfg = config.get("robot_overlay", {})
    return {
        "enabled": bool(overlay_cfg.get("enabled", True)),
        "length_m": float(overlay_cfg.get("length_m", 1.20)),
        "width_m": float(overlay_cfg.get("width_m", 0.80)),
        "center_offset_x_m": float(overlay_cfg.get("center_offset_x_m", 0.0)),
        "center_offset_z_m": float(overlay_cfg.get("center_offset_z_m", 0.0)),
        "height_offset_y_m": float(overlay_cfg.get("height_offset_y_m", 0.05)),
        "heading_len_m": float(overlay_cfg.get("heading_len_m", 0.45)),
    }


def build_robot_overlay_world_from_pose(pose_4x4, options):
    mat = _to_np_4x4(pose_4x4)
    origin = np.array([float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])], dtype=np.float64)
    # Keep overlay planar to avoid roll/pitch distortion.
    right = np.array([float(mat[0, 0]), 0.0, float(mat[2, 0])], dtype=np.float64)
    forward = np.array([-float(mat[0, 2]), 0.0, -float(mat[2, 2])], dtype=np.float64)
    nr = float(np.linalg.norm(right))
    nf = float(np.linalg.norm(forward))
    if nr < 1e-9:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / nr
    if nf < 1e-9:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        forward = forward / nf

    length_m = max(0.10, float(options.get("length_m", 1.20)))
    width_m = max(0.10, float(options.get("width_m", 0.80)))
    hl = length_m * 0.5
    hw = width_m * 0.5
    center = origin + (right * float(options.get("center_offset_x_m", 0.0))) + (forward * float(options.get("center_offset_z_m", 0.0)))
    y = float(origin[1]) + float(options.get("height_offset_y_m", 0.05))

    fl = center + (forward * hl) - (right * hw)
    fr = center + (forward * hl) + (right * hw)
    rr = center - (forward * hl) + (right * hw)
    rl = center - (forward * hl) - (right * hw)
    body_points_world = [
        float(fl[0]), y, float(fl[2]),
        float(fr[0]), y, float(fr[2]),
        float(rr[0]), y, float(rr[2]),
        float(rl[0]), y, float(rl[2]),
    ]

    heading_len = max(0.10, float(options.get("heading_len_m", 0.45)))
    tip = center + (forward * (hl + heading_len))
    base_center = center + (forward * (hl * 0.55))
    half_base = max(0.05, width_m * 0.18)
    left = base_center - (right * half_base)
    right_pt = base_center + (right * half_base)
    heading_points_world = [
        float(tip[0]), y, float(tip[2]),
        float(right_pt[0]), y, float(right_pt[2]),
        float(left[0]), y, float(left[2]),
    ]
    return body_points_world, heading_points_world


def setup_console_input():
    if os.name == "nt":
        return None
    if not sys.stdin.isatty():
        return None
    try:
        fd = sys.stdin.fileno()
        old_termios = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return {"fd": fd, "old_termios": old_termios}
    except Exception as e:
        print(f"[Input] Failed to enable non-blocking console input: {e}")
        return None


def restore_console_input(state):
    if os.name == "nt" or not state:
        return
    try:
        termios.tcsetattr(state["fd"], termios.TCSADRAIN, state["old_termios"])
    except Exception:
        pass


def poll_console_key(state):
    if not state:
        return None
    try:
        ready, _, _ = select.select([state["fd"]], [], [], 0)
        if not ready:
            return None
        return os.read(state["fd"], 1)
    except Exception:
        return None


def load_config_json(config_path):
    config_abs_path = os.path.abspath(config_path)
    print(f"[Config] Loading config: {config_abs_path}")

    if not os.path.exists(config_path):
        print(f"[Config] Config not found: {config_path} -> using defaults")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Config] Failed to read config {config_path}: {e} -> using defaults")
        return {}


def load_display_options(config):
    display_cfg = config.get("display", {})
    return {
        "pc_window_enabled": bool(display_cfg.get("pc_window_enabled", True)),
    }


def load_web_options(config):
    web_cfg = config.get("web", {})
    return {
        "enabled": bool(web_cfg.get("enabled", False)),
        "host": str(web_cfg.get("host", "0.0.0.0")),
        "port": int(web_cfg.get("port", 8080)),
    }


def load_calibration_options(config):
    calib_cfg = config.get("calibration", {})
    return {
        "max_distance_m": float(calib_cfg.get("max_distance_m", 4.0)),
        "vertical_range_m": float(calib_cfg.get("vertical_range_m", 0.35)),
        "angle_min_deg": float(calib_cfg.get("angle_min_deg", -55.0)),
        "angle_max_deg": float(calib_cfg.get("angle_max_deg", 230.0)),
    }


def load_zed_options(config):
    zed_cfg = config.get("zed", {})
    init_cfg = zed_cfg.get("init", {})
    tracking_cfg = zed_cfg.get("tracking", {})
    mapping_cfg = zed_cfg.get("mapping", {})
    runtime_cfg = zed_cfg.get("runtime", {})

    return {
        "init": {
            "depth_mode": str(init_cfg.get("depth_mode", "NEURAL")),
            "coordinate_units": str(init_cfg.get("coordinate_units", "METER")),
            "coordinate_system": str(init_cfg.get("coordinate_system", "RIGHT_HANDED_Y_UP")),
            # None means "AUTO / SDK default".
            "depth_minimum_distance": init_cfg.get("depth_minimum_distance", None),
            "depth_maximum_distance": float(init_cfg.get("depth_maximum_distance", 10.0)),
            # None means "do not override SDK default FPS".
            "camera_fps": init_cfg.get("camera_fps", None),
        },
        "tracking": {
            "enable_area_memory": bool(tracking_cfg.get("enable_area_memory", False)),
            "mode": tracking_cfg.get("mode", None),
            "initial_position_m": {
                "x": float(tracking_cfg.get("initial_position_m", {}).get("x", 0.0)),
                "y": float(tracking_cfg.get("initial_position_m", {}).get("y", 0.30)),
                "z": float(tracking_cfg.get("initial_position_m", {}).get("z", 0.0)),
            },
        },
        "mapping": {
            "map_type": str(mapping_cfg.get("map_type", "MESH")),
            "save_texture": bool(mapping_cfg.get("save_texture", True)),
            # preset string ("LOW/MEDIUM/HIGH") or numeric meter
            "resolution": mapping_cfg.get("resolution", "MEDIUM"),
            # preset string ("SHORT/MEDIUM/LONG") or numeric meter
            "range": mapping_cfg.get("range", "MEDIUM"),
            "use_chunk_only": bool(mapping_cfg.get("use_chunk_only", True)),
            # Optional advanced params (SDK defaults when None)
            "max_memory_usage": mapping_cfg.get("max_memory_usage", None),
            "stability_counter": mapping_cfg.get("stability_counter", None),
        },
        "runtime": {
            # None means "do not override SDK default"
            "measure3D_reference_frame": runtime_cfg.get("measure3D_reference_frame", None),
        },
    }


def _enum_value(enum_owner, name, default_value=None, label="enum"):
    if name is None:
        return default_value
    key = str(name).strip()
    try:
        return getattr(enum_owner, key)
    except Exception:
        print(f"[Config] Invalid {label} '{key}' -> fallback")
        return default_value


def load_lidar_alert_options(config):
    alert_cfg = config.get("lidar_2d_alert_threshold", {})
    return {
        "enabled": bool(alert_cfg.get("enabled", False)),
        "min_m": float(alert_cfg.get("min_m", 0.0)),
        "max_m": float(alert_cfg.get("max_m", 1.0)),
    }


def load_lidar_receivers(config_path, config=None):
    config_abs_path = os.path.abspath(config_path)
    print(f"[LiDAR] Using config: {config_abs_path}")

    # Backward-compatible default: single LiDAR if config is absent.
    if config is None:
        config = load_config_json(config_path)

    lidar_items = config.get("lidars", [])
    alert_options = load_lidar_alert_options(config)
    print(
        f"[LiDAR] 2D alert threshold: enabled={alert_options['enabled']} "
        f"range=[{alert_options['min_m']:.2f}, {alert_options['max_m']:.2f}] m"
    )
    receivers = []

    for idx, item in enumerate(lidar_items):
        if not item.get("enabled", True):
            continue

        name = item.get("name", f"lidar_{idx + 1}")
        ip = item.get("ip", "192.168.0.31")
        port = int(item.get("port", 8000))

        offset = item.get("offset", {})
        offset_x = float(offset.get("x", item.get("offset_x", -0.12)))
        offset_y = float(offset.get("y", item.get("offset_y", 0.0)))
        offset_z = float(offset.get("z", item.get("offset_z", 0.0)))
        rotation = item.get("rotation", {})
        yaw_deg = float(rotation.get("yaw_deg", item.get("yaw_deg", 0.0)))

        receiver = lidar_thread.LidarReceiver(
            ip=ip,
            port=port,
            name=name,
            offset_x=offset_x,
            offset_y=offset_y,
            offset_z=offset_z,
            yaw_deg=yaw_deg,
            alert_enabled=alert_options["enabled"],
            alert_min_m=alert_options["min_m"],
            alert_max_m=alert_options["max_m"],
        )
        receivers.append(receiver)

    if not receivers:
        print(f"[LiDAR] No enabled lidars in {config_path} -> fallback to single default lidar")
        receivers.append(lidar_thread.LidarReceiver())
    else:
        names = ", ".join([f"{r.name}({r.ip}:{r.port})" for r in receivers])
        print(f"[LiDAR] Enabled receivers: {names}")

    return receivers


def reset_spatial_mapping_session(zed, viewer, pymesh, mapping_params):
    # Follow ZED spatial mapping sample reset flow:
    # reset tracking pose + clear current mesh buffers + enable mapping again.
    try:
        zed.disable_spatial_mapping()
    except Exception:
        pass

    init_pose = sl.Transform()
    init_pose.set_identity()
    zed.reset_positional_tracking(init_pose)

    if pymesh is not None:
        pymesh.clear()
    if viewer is not None:
        viewer.clear_current_mesh()

    err = zed.enable_spatial_mapping(mapping_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Enable spatial mapping failed after reset: {err}")


def print_zed_settings_snapshot(tag, init_params, tracking_params, mapping_params, runtime_params):
    camera_fps = getattr(init_params, "camera_fps", None)
    depth_max = getattr(init_params, "depth_maximum_distance", None)
    enable_area_memory = getattr(tracking_params, "enable_area_memory", None)
    tracking_mode = getattr(tracking_params, "mode", None)
    map_type = getattr(mapping_params, "map_type", None)
    resolution_meter = getattr(mapping_params, "resolution_meter", None)
    range_meter = getattr(mapping_params, "range_meter", None)
    use_chunk_only = getattr(mapping_params, "use_chunk_only", None)
    save_texture = getattr(mapping_params, "save_texture", None)
    measure_ref = getattr(runtime_params, "measure3D_reference_frame", None)

    print(f"[ZED {tag}] [1] camera_fps={camera_fps if camera_fps is not None else 'SDK default'}")
    print(f"[ZED {tag}] [2] depth_maximum_distance={depth_max if depth_max is not None else 'SDK default'}")
    print(
        f"[ZED {tag}] [3] positional_tracking: "
        f"enable_area_memory={enable_area_memory if enable_area_memory is not None else 'SDK default'}, "
        f"mode={tracking_mode if tracking_mode is not None else 'SDK default'}"
    )
    print(
        f"[ZED {tag}] [5] spatial_mapping: "
        f"map_type={map_type if map_type is not None else 'SDK default'}, "
        f"resolution_meter={resolution_meter if resolution_meter is not None else 'SDK default'}, "
        f"range_meter={range_meter if range_meter is not None else 'SDK default'}, "
        f"use_chunk_only={use_chunk_only if use_chunk_only is not None else 'SDK default'}, "
        f"save_texture={save_texture if save_texture is not None else 'SDK default'}"
    )
    print(
        f"[ZED {tag}] [6] runtime.measure3D_reference_frame="
        f"{measure_ref if measure_ref is not None else 'SDK default'}"
    )


def main():
    server = None
    viewer = None
    zed = None
    image = None
    pymesh = None
    lidars = []
    config_path = os.path.abspath(CONFIG_PATH)
    config = load_config_json(config_path)
    display_options = load_display_options(config)
    web_options = load_web_options(config)
    zed_options = load_zed_options(config)
    pc_window_enabled = display_options["pc_window_enabled"]
    web_enabled = bool(web_options["enabled"])
    web_host = web_options["host"]
    web_port = web_options["port"]
    calib_options = load_calibration_options(config)
    offset_ui_state = {
        "selected_idx": 0,
        "step": 0.01,  # meter
        "yaw_step_deg": 0.5,  # degree
    }
    alert_ui_state = load_lidar_alert_options(config)
    robot_overlay_options = load_robot_overlay_options(config)
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
    pending_config_save = False
    last_config_save_time = 0.0
    config_save_interval_sec = 0.5
    console_input_state = None
    save_map_requested = False
    prev_pose_time_s = None
    prev_translation = None
    velocity_alpha = 0.2
    robot_velocity_state = {
        "valid": False,
        "vx_mps": 0.0,
        "vy_mps": 0.0,
        "vz_mps": 0.0,
        "speed_mps": 0.0,
        "speed_kph": 0.0,
    }
    prev_grab_wall_time = None
    last_map_retrieve_time = None
    pose_history = deque(maxlen=180)
    runtime_perf = {
        "zed_fps": 0.0,
        "map_hz": 0.0,
        "lidar_pose_lag_ms": 0.0,
    }
    calib_map_points = np.empty((0, 3), dtype=np.float32)
    calib_last_compute_s = 0.0
    calib_compute_interval_s = 0.35
    calib_max_map_points = 50000
    calib_max_lidar_points = 5000
    calib_rmse_alpha = 0.2
    calib_state = {
        "active": False,
        "selected_name": "",
        "rmse_m": None,
        "rmse_ema_m": None,
        "sample_count": 0,
        "map_point_count": 0,
        "note": "idle",
        "updated_at_s": None,
    }
    config_sync_state = {
        "path": str(config_path),
        "last_loaded_s": time.time(),
        "last_saved_s": None,
        "last_action": "startup_load",
        "last_error": "",
    }

    data_logger = DataLogger()

    try:
        console_input_state = setup_console_input()
        # Print SDK-default snapshots before any explicit program override.
        default_init = sl.InitParameters()
        default_tracking = sl.PositionalTrackingParameters()
        default_mapping = sl.SpatialMappingParameters()
        default_runtime = sl.RuntimeParameters()
        print_zed_settings_snapshot("Default", default_init, default_tracking, default_mapping, default_runtime)

        # 1. Initialize ZED
        print("Initializing ZED Camera...")
        init = sl.InitParameters()
        init.depth_mode = _enum_value(
            sl.DEPTH_MODE,
            zed_options["init"]["depth_mode"],
            default_value=sl.DEPTH_MODE.NEURAL,
            label="zed.init.depth_mode",
        )
        init.coordinate_units = _enum_value(
            sl.UNIT,
            zed_options["init"]["coordinate_units"],
            default_value=sl.UNIT.METER,
            label="zed.init.coordinate_units",
        )
        init.coordinate_system = _enum_value(
            sl.COORDINATE_SYSTEM,
            zed_options["init"]["coordinate_system"],
            default_value=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
            label="zed.init.coordinate_system",
        )
        init.depth_maximum_distance = float(zed_options["init"]["depth_maximum_distance"])
        if zed_options["init"]["depth_minimum_distance"] is not None:
            init.depth_minimum_distance = float(zed_options["init"]["depth_minimum_distance"])
        if zed_options["init"]["camera_fps"] is not None:
            init.camera_fps = int(zed_options["init"]["camera_fps"])
        
        zed = sl.Camera()
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Camera Open Failed: {status}")
            return
        try:
            cam_info = zed.get_camera_information()
            cam_cfg = cam_info.camera_configuration
            resolution = cam_cfg.resolution
            fps = cam_cfg.fps
            serial = getattr(cam_info, "serial_number", None)
            if serial is None:
                serial = getattr(cam_cfg, "serial_number", "unknown")
            model = getattr(cam_info, "camera_model", "unknown")
            print(
                f"[ZED] Model={model} Serial={serial} "
                f"Resolution={resolution.width}x{resolution.height} FPS={fps}"
            )
        except Exception as e:
            print(f"[ZED] Model info unavailable: {e}")

        # 2. Enable Tracking & Mapping
        print("Enabling Tracking and Spatial Mapping...")
        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.enable_area_memory = bool(zed_options["tracking"]["enable_area_memory"])
        tracking_mode = _enum_value(
            sl.POSITIONAL_TRACKING_MODE,
            zed_options["tracking"]["mode"],
            default_value=None,
            label="zed.tracking.mode",
        )
        if tracking_mode is not None:
            tracking_params.mode = tracking_mode
        
        # Set initial position (Camera at 30cm height)
        initial_position = sl.Transform()
        initial_translation = sl.Translation()
        initial_translation.init_vector(
            float(zed_options["tracking"]["initial_position_m"]["x"]),
            float(zed_options["tracking"]["initial_position_m"]["y"]),
            float(zed_options["tracking"]["initial_position_m"]["z"]),
        ) # x, y, z
        initial_position.set_translation(initial_translation)
        tracking_params.set_initial_world_transform(initial_position)
        
        tracking_err = zed.enable_positional_tracking(tracking_params)
        if tracking_err != sl.ERROR_CODE.SUCCESS:
            print(f"Enable positional tracking failed: {tracking_err}")
            return
        
        mapping_params = sl.SpatialMappingParameters()
        mapping_params.map_type = _enum_value(
            sl.SPATIAL_MAP_TYPE,
            zed_options["mapping"]["map_type"],
            default_value=sl.SPATIAL_MAP_TYPE.MESH,
            label="zed.mapping.map_type",
        )
        mapping_params.save_texture = bool(zed_options["mapping"]["save_texture"])
        mapping_params.use_chunk_only = bool(zed_options["mapping"]["use_chunk_only"])
        if zed_options["mapping"]["max_memory_usage"] is not None:
            mapping_params.max_memory_usage = int(zed_options["mapping"]["max_memory_usage"])
        if zed_options["mapping"]["stability_counter"] is not None:
            mapping_params.stability_counter = int(zed_options["mapping"]["stability_counter"])

        res_cfg = zed_options["mapping"]["resolution"]
        if isinstance(res_cfg, str):
            res_preset = _enum_value(
                sl.MAPPING_RESOLUTION,
                res_cfg,
                default_value=sl.MAPPING_RESOLUTION.MEDIUM,
                label="zed.mapping.resolution",
            )
            mapping_params.resolution_meter = mapping_params.get_resolution_preset(res_preset)
        else:
            mapping_params.resolution_meter = float(res_cfg)

        range_cfg = zed_options["mapping"]["range"]
        if isinstance(range_cfg, str):
            range_preset = _enum_value(
                sl.MAPPING_RANGE,
                range_cfg,
                default_value=sl.MAPPING_RANGE.MEDIUM,
                label="zed.mapping.range",
            )
            mapping_params.range_meter = mapping_params.get_range_preset(range_preset)
        else:
            mapping_params.range_meter = float(range_cfg)
        
        # 3. Start LiDAR Threads
        lidars = load_lidar_receivers(config_path, config=config)
        print(f"Starting LiDAR Receivers... count={len(lidars)}")
        for lidar in lidars:
            lidar.start()

        # 4. Initialize Viewer
        camera_info = zed.get_camera_information()
        viewer = gl.GLViewer()
        draw_mesh_mode = (mapping_params.map_type == sl.SPATIAL_MAP_TYPE.MESH)
        if draw_mesh_mode:
            pymesh = sl.Mesh()
        else:
            pymesh = sl.FusedPointCloud()
        viewer.init(
            camera_info.camera_configuration.calibration_parameters.left_cam,
            pymesh,
            draw_mesh_mode,
            show_window=pc_window_enabled,
        )
        print(f"[Display] PC window enabled: {pc_window_enabled}")

        def get_selected_lidar():
            if not lidars:
                return None
            idx = max(0, min(int(offset_ui_state["selected_idx"]), len(lidars) - 1))
            offset_ui_state["selected_idx"] = idx
            return lidars[idx]

        def get_runtime_lidar_state():
            items = []
            for lidar in lidars:
                status = lidar.get_status()
                items.append({
                    "name": lidar.name,
                    "connected": bool(status.get("connected", False)),
                    "fps": float(status.get("fps", 0.0)),
                    "point_count": int(status.get("point_count", 0)),
                    "offset": status.get("offset", {"x": 0.0, "y": 0.0, "z": 0.0}),
                    "yaw_deg": float(status.get("yaw_deg", 0.0)),
                    "alert_threshold": status.get("alert_threshold", {"enabled": False, "min_m": 0.0, "max_m": 1.0}),
                })
            selected_name = None
            selected = get_selected_lidar()
            if selected is not None:
                selected_name = selected.name
            return {
                "selected_name": selected_name,
                "step": float(offset_ui_state["step"]),
                "yaw_step_deg": float(offset_ui_state["yaw_step_deg"]),
                "alert_threshold": {
                    "enabled": bool(alert_ui_state["enabled"]),
                    "min_m": float(alert_ui_state["min_m"]),
                    "max_m": float(alert_ui_state["max_m"]),
                },
                "robot_velocity": dict(robot_velocity_state),
                "is_logging": bool(data_logger.is_logging),
                "profiles": sorted([str(k) for k in profiles.keys()]),
                "calibration": dict(calib_state),
                "config_sync": dict(config_sync_state),
                "lidars": items,
            }

        def update_control_status():
            selected = get_selected_lidar()
            speed_mps = float(robot_velocity_state.get("speed_mps", 0.0))
            speed_kph = float(robot_velocity_state.get("speed_kph", 0.0))
            text = f"v={speed_mps:.2f}m/s({speed_kph:.2f}km/h)"
            if viewer is not None:
                viewer.set_control_status(text)

        def get_pose_nearest_to_time(frame_time_s, default_t_world_robot):
            if frame_time_s is None or not pose_history:
                return default_t_world_robot, None
            best_t = None
            best_mat = None
            best_abs = None
            for t_s, mat in pose_history:
                dt = abs(float(t_s) - float(frame_time_s))
                if best_abs is None or dt < best_abs:
                    best_abs = dt
                    best_t = float(t_s)
                    best_mat = mat
            if best_mat is None:
                return default_t_world_robot, None
            lag_ms = (float(best_t) - float(frame_time_s)) * 1000.0
            return best_mat, lag_ms

        def update_robot_velocity_from_pose(cur_pose):
            nonlocal prev_pose_time_s, prev_translation
            try:
                t_s = float(cur_pose.timestamp.get_milliseconds()) * 0.001
                tr = cur_pose.get_translation(sl.Translation()).get()
                cur_translation = np.array([
                    float(tr[0]),
                    float(tr[1]),
                    float(tr[2]),
                ], dtype=np.float64)
            except Exception:
                return

            if prev_pose_time_s is not None and prev_translation is not None:
                dt = t_s - prev_pose_time_s
                if dt > 1e-6:
                    v = (cur_translation - prev_translation) / dt
                    vx = float(v[0])
                    vy = float(v[1])
                    vz = float(v[2])
                    speed = float(math.sqrt((vx * vx) + (vy * vy) + (vz * vz)))

                    if not robot_velocity_state["valid"]:
                        robot_velocity_state["vx_mps"] = vx
                        robot_velocity_state["vy_mps"] = vy
                        robot_velocity_state["vz_mps"] = vz
                    else:
                        a = float(velocity_alpha)
                        robot_velocity_state["vx_mps"] = ((1.0 - a) * robot_velocity_state["vx_mps"]) + (a * vx)
                        robot_velocity_state["vy_mps"] = ((1.0 - a) * robot_velocity_state["vy_mps"]) + (a * vy)
                        robot_velocity_state["vz_mps"] = ((1.0 - a) * robot_velocity_state["vz_mps"]) + (a * vz)
                    sx = float(robot_velocity_state["vx_mps"])
                    sy = float(robot_velocity_state["vy_mps"])
                    sz = float(robot_velocity_state["vz_mps"])
                    sm = float(math.sqrt((sx * sx) + (sy * sy) + (sz * sz)))
                    robot_velocity_state["speed_mps"] = sm
                    robot_velocity_state["speed_kph"] = sm * 3.6
                    robot_velocity_state["valid"] = True

            prev_pose_time_s = t_s
            prev_translation = cur_translation

        def rebuild_calib_map_points():
            nonlocal calib_map_points
            pts_parts = []
            try:
                chunks = getattr(pymesh, "chunks", [])
            except Exception:
                chunks = []
            for ch in chunks:
                try:
                    verts = np.asarray(ch.vertices, dtype=np.float32)
                except Exception:
                    continue
                if verts.size == 0:
                    continue
                if verts.ndim == 1:
                    if (verts.size % 3) != 0:
                        continue
                    xyz = verts.reshape(-1, 3)
                else:
                    xyz = verts.reshape(verts.shape[0], -1)[:, :3]
                if xyz.shape[0] <= 0:
                    continue
                stride = max(1, int(xyz.shape[0] / 5000))
                pts_parts.append(xyz[::stride])
            if not pts_parts:
                calib_map_points = np.empty((0, 3), dtype=np.float32)
                calib_state["map_point_count"] = 0
                return
            merged = np.concatenate(pts_parts, axis=0)
            if merged.shape[0] > calib_max_map_points:
                stride = max(1, int(merged.shape[0] / calib_max_map_points))
                merged = merged[::stride]
            calib_map_points = np.asarray(merged, dtype=np.float32)
            calib_state["map_point_count"] = int(calib_map_points.shape[0])

        def compute_rmse_points_to_map_local(points_local_flat, t_world_lidar):
            if calib_map_points.shape[0] < 64:
                return None, 0
            if not points_local_flat:
                return None, 0
            pts = np.asarray(points_local_flat, dtype=np.float32).reshape(-1, 3)
            if pts.shape[0] > calib_max_lidar_points:
                stride = max(1, int(pts.shape[0] / calib_max_lidar_points))
                pts = pts[::stride]

            # Build map points in selected LiDAR local frame, then compare in local ROI.
            try:
                t_inv = np.linalg.inv(np.asarray(t_world_lidar, dtype=np.float64))
            except Exception:
                return None, 0
            map_w = np.asarray(calib_map_points, dtype=np.float64)
            ones = np.ones((map_w.shape[0], 1), dtype=np.float64)
            map_h = np.concatenate([map_w, ones], axis=1)
            map_local = (t_inv @ map_h.T).T[:, :3].astype(np.float32)

            # Parameters from config
            max_r = calib_options["max_distance_m"]
            vert_h = calib_options["vertical_range_m"]
            a_min = calib_options["angle_min_deg"]
            a_max = calib_options["angle_max_deg"]

            # LiDAR FOV-like ROI in local frame improves sensitivity to x/y/z/yaw tuning.
            mr = np.sqrt((map_local[:, 0] * map_local[:, 0]) + (map_local[:, 2] * map_local[:, 2]))
            ma = np.degrees(np.arctan2(-map_local[:, 0], -map_local[:, 2]))
            map_mask = (
                (mr > 0.05) & (mr < max_r) &
                (np.abs(map_local[:, 1]) < vert_h) &
                (ma >= a_min) & (ma <= a_max)
            )
            model = map_local[map_mask]
            if model.shape[0] < 64:
                return None, int(model.shape[0])

            pr = np.sqrt((pts[:, 0] * pts[:, 0]) + (pts[:, 2] * pts[:, 2]))
            pa = np.degrees(np.arctan2(-pts[:, 0], -pts[:, 2]))
            pts_mask = (
                (pr > 0.05) & (pr < max_r) &
                (np.abs(pts[:, 1]) < vert_h) &
                (pa >= a_min) & (pa <= a_max)
            )
            pts = pts[pts_mask]
            if pts.shape[0] < 16:
                return None, int(pts.shape[0])

            if HAS_SCIPY:
                tree = cKDTree(model)
                distances, _ = tree.query(pts, k=1)
                nearest = distances
            else:
                # Brute-force nearest distance (bounded point counts for realtime).
                diff = pts[:, None, :] - model[None, :, :]
                d2 = np.sum(diff * diff, axis=2)
                nearest = np.sqrt(np.min(d2, axis=1))
                nearest = nearest[np.isfinite(nearest)]

            if nearest.size < 16:
                return None, int(nearest.size)
            valid = nearest[(nearest > 0.005) & (nearest < 0.8)]
            if valid.size < 16:
                return None, int(valid.size)
            p80 = float(np.percentile(valid, 80))
            inlier = valid[valid <= p80]
            if inlier.size < 16:
                return None, int(inlier.size)
            rmse = float(np.sqrt(np.mean(inlier * inlier)))
            return rmse, int(inlier.size)

        def persist_profiles():
            nonlocal config
            try:
                disk_cfg = {}
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            disk_cfg = json.load(f)
                            if not isinstance(disk_cfg, dict):
                                disk_cfg = {}
                    except Exception:
                        disk_cfg = {}
                if not disk_cfg:
                    disk_cfg = config if isinstance(config, dict) else {}
                if not isinstance(disk_cfg, dict):
                    disk_cfg = {}

                disk_cfg["profiles"] = profiles
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(disk_cfg, f, indent=2)
                config = disk_cfg
                return True
            except Exception as e:
                print(f"[Profile] Failed to save profiles to {config_path}: {e}")
                return False

        def request_config_save():
            nonlocal pending_config_save
            pending_config_save = True

        def persist_lidar_config():
            nonlocal config
            try:
                # Reload current file to preserve unrelated sections edited externally.
                disk_cfg = {}
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            disk_cfg = json.load(f)
                            if not isinstance(disk_cfg, dict):
                                disk_cfg = {}
                    except Exception:
                        disk_cfg = {}
                if not disk_cfg:
                    disk_cfg = config if isinstance(config, dict) else {}
                if not isinstance(disk_cfg, dict):
                    disk_cfg = {}

                items = disk_cfg.get("lidars", [])
                if not isinstance(items, list):
                    items = []

                for lidar in lidars:
                    status = lidar.get_status()
                    off = status.get("offset", {"x": 0.0, "y": 0.0, "z": 0.0})
                    yaw_deg = float(status.get("yaw_deg", 0.0))

                    entry = next((x for x in items if str(x.get("name", "")) == lidar.name), None)
                    if entry is None:
                        entry = {
                            "name": lidar.name,
                            "enabled": True,
                            "ip": lidar.ip,
                            "port": int(lidar.port),
                        }
                        items.append(entry)

                    entry["offset"] = {
                        "x": float(off.get("x", 0.0)),
                        "y": float(off.get("y", 0.0)),
                        "z": float(off.get("z", 0.0)),
                    }
                    rotation = entry.get("rotation", {})
                    if not isinstance(rotation, dict):
                        rotation = {}
                    rotation["yaw_deg"] = yaw_deg
                    entry["rotation"] = rotation

                disk_cfg["lidars"] = items
                disk_cfg["lidar_2d_alert_threshold"] = {
                    "enabled": bool(alert_ui_state["enabled"]),
                    "min_m": float(alert_ui_state["min_m"]),
                    "max_m": float(alert_ui_state["max_m"]),
                }
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(disk_cfg, f, indent=2)
                config = disk_cfg
                config_sync_state["last_saved_s"] = time.time()
                config_sync_state["last_action"] = "saved_to_disk"
                config_sync_state["last_error"] = ""
                return True
            except Exception as e:
                print(f"[Config] Failed to save lidar config {config_path}: {e}")
                config_sync_state["last_error"] = str(e)
                config_sync_state["last_action"] = "save_failed"
                return False

        def reload_lidar_config_from_disk():
            nonlocal config
            try:
                disk_cfg = load_config_json(config_path)
                if not isinstance(disk_cfg, dict):
                    disk_cfg = {}
                # Apply alert threshold from disk.
                alert_cfg = disk_cfg.get("lidar_2d_alert_threshold", {})
                if isinstance(alert_cfg, dict):
                    alert_ui_state["enabled"] = bool(alert_cfg.get("enabled", alert_ui_state["enabled"]))
                    alert_ui_state["min_m"] = float(alert_cfg.get("min_m", alert_ui_state["min_m"]))
                    alert_ui_state["max_m"] = float(alert_cfg.get("max_m", alert_ui_state["max_m"]))
                    if alert_ui_state["min_m"] > alert_ui_state["max_m"]:
                        alert_ui_state["min_m"], alert_ui_state["max_m"] = alert_ui_state["max_m"], alert_ui_state["min_m"]
                    for lidar in lidars:
                        lidar.set_alert_threshold(
                            enabled=alert_ui_state["enabled"],
                            min_m=alert_ui_state["min_m"],
                            max_m=alert_ui_state["max_m"],
                        )

                # Apply offsets/yaw by lidar name from disk.
                items = disk_cfg.get("lidars", [])
                if not isinstance(items, list):
                    items = []
                by_name = {}
                for one in items:
                    if not isinstance(one, dict):
                        continue
                    name = str(one.get("name", "")).strip()
                    if name:
                        by_name[name] = one
                for lidar in lidars:
                    one = by_name.get(lidar.name, {})
                    off = one.get("offset", {})
                    if not isinstance(off, dict):
                        off = {}
                    rot = one.get("rotation", {})
                    if not isinstance(rot, dict):
                        rot = {}
                    lidar.set_offset(
                        x=off.get("x", None),
                        y=off.get("y", None),
                        z=off.get("z", None),
                    )
                    if "yaw_deg" in rot:
                        lidar.set_yaw_deg(float(rot.get("yaw_deg", 0.0)))

                config = disk_cfg
                config_sync_state["last_loaded_s"] = time.time()
                config_sync_state["last_action"] = "reloaded_from_disk"
                config_sync_state["last_error"] = ""
                print(f"[Config] Reloaded from disk: {config_path}")
                return True
            except Exception as e:
                print(f"[Config] Failed to reload config {config_path}: {e}")
                config_sync_state["last_error"] = str(e)
                config_sync_state["last_action"] = "reload_failed"
                return False

        def apply_offset_control(action, payload):
            if not lidars:
                return False

            if action == "select_prev_lidar":
                offset_ui_state["selected_idx"] = (int(offset_ui_state["selected_idx"]) - 1) % len(lidars)
                selected = get_selected_lidar()
                if selected is not None:
                    print(f"[Offset] selected lidar: {selected.name}")
                return True
            if action == "select_next_lidar":
                offset_ui_state["selected_idx"] = (int(offset_ui_state["selected_idx"]) + 1) % len(lidars)
                selected = get_selected_lidar()
                if selected is not None:
                    print(f"[Offset] selected lidar: {selected.name}")
                return True
            if action == "select_lidar_index":
                idx = int(payload.get("index", -1))
                if idx < 0 or idx >= len(lidars):
                    return False
                offset_ui_state["selected_idx"] = idx
                selected = get_selected_lidar()
                if selected is not None:
                    print(f"[Offset] selected lidar: {selected.name}")
                return True

            if action == "offset_step_down":
                offset_ui_state["step"] = max(0.001, float(offset_ui_state["step"]) * 0.5)
                print(f"[Offset] step: {offset_ui_state['step']:.4f} m")
                return True
            if action == "offset_step_up":
                offset_ui_state["step"] = min(1.0, float(offset_ui_state["step"]) * 2.0)
                print(f"[Offset] step: {offset_ui_state['step']:.4f} m")
                return True
            if action == "offset_set_step":
                step = float(payload.get("step", offset_ui_state["step"]))
                offset_ui_state["step"] = max(0.001, min(1.0, step))
                print(f"[Offset] step: {offset_ui_state['step']:.4f} m")
                return True
            if action == "yaw_step_down":
                offset_ui_state["yaw_step_deg"] = max(0.1, float(offset_ui_state["yaw_step_deg"]) * 0.5)
                print(f"[Yaw] step: {offset_ui_state['yaw_step_deg']:.2f} deg")
                return True
            if action == "yaw_step_up":
                offset_ui_state["yaw_step_deg"] = min(20.0, float(offset_ui_state["yaw_step_deg"]) * 2.0)
                print(f"[Yaw] step: {offset_ui_state['yaw_step_deg']:.2f} deg")
                return True
            if action == "yaw_set_step":
                step = float(payload.get("yaw_step_deg", offset_ui_state["yaw_step_deg"]))
                offset_ui_state["yaw_step_deg"] = max(0.1, min(20.0, step))
                print(f"[Yaw] step: {offset_ui_state['yaw_step_deg']:.2f} deg")
                return True
            if action == "set_step_preset":
                mode = str(payload.get("mode", ""))
                if mode == "fine":
                    offset_ui_state["step"] = 0.002
                    offset_ui_state["yaw_step_deg"] = 0.2
                elif mode == "normal":
                    offset_ui_state["step"] = 0.01
                    offset_ui_state["yaw_step_deg"] = 0.5
                elif mode == "coarse":
                    offset_ui_state["step"] = 0.05
                    offset_ui_state["yaw_step_deg"] = 1.0
                else:
                    return False
                print(f"[Adjust] preset={mode} step={offset_ui_state['step']:.3f}m yaw_step={offset_ui_state['yaw_step_deg']:.2f}deg")
                return True
            if action == "alert_threshold_set":
                enabled = payload.get("enabled", alert_ui_state["enabled"])
                min_m = float(payload.get("min_m", alert_ui_state["min_m"]))
                max_m = float(payload.get("max_m", alert_ui_state["max_m"]))
                if min_m > max_m:
                    min_m, max_m = max_m, min_m
                alert_ui_state["enabled"] = bool(enabled)
                alert_ui_state["min_m"] = float(min_m)
                alert_ui_state["max_m"] = float(max_m)
                for lidar in lidars:
                    lidar.set_alert_threshold(
                        enabled=alert_ui_state["enabled"],
                        min_m=alert_ui_state["min_m"],
                        max_m=alert_ui_state["max_m"],
                    )
                request_config_save()
                print(
                    f"[Alert] threshold enabled={alert_ui_state['enabled']} "
                    f"range=[{alert_ui_state['min_m']:.2f}, {alert_ui_state['max_m']:.2f}] m"
                )
                return True

            selected = get_selected_lidar()
            if selected is None:
                return False

            step = float(offset_ui_state["step"])
            changed_calib = False
            if action == "offset_x_minus":
                selected.add_offset(dx=-step)
                changed_calib = True
            elif action == "offset_x_plus":
                selected.add_offset(dx=step)
                changed_calib = True
            elif action == "offset_y_minus":
                selected.add_offset(dy=-step)
                changed_calib = True
            elif action == "offset_y_plus":
                selected.add_offset(dy=step)
                changed_calib = True
            elif action == "offset_z_minus":
                selected.add_offset(dz=-step)
                changed_calib = True
            elif action == "offset_z_plus":
                selected.add_offset(dz=step)
                changed_calib = True
            elif action == "reset_selected_lidar_offset":
                selected.set_offset(x=0.0, y=0.0, z=0.0)
                changed_calib = True
            elif action == "yaw_minus":
                selected.add_yaw_deg(-float(offset_ui_state["yaw_step_deg"]))
                changed_calib = True
            elif action == "yaw_plus":
                selected.add_yaw_deg(float(offset_ui_state["yaw_step_deg"]))
                changed_calib = True
            elif action == "reset_selected_lidar_yaw":
                selected.set_yaw_deg(0.0)
                changed_calib = True
            elif action == "lidar_offset_delta":
                name = str(payload.get("name", ""))
                axis = str(payload.get("axis", "")).lower()
                delta = float(payload.get("delta", 0.0))
                target = next((l for l in lidars if l.name == name), None)
                if target is None:
                    return False
                if axis == "x":
                    target.add_offset(dx=delta)
                elif axis == "y":
                    target.add_offset(dy=delta)
                elif axis == "z":
                    target.add_offset(dz=delta)
                else:
                    return False
                request_config_save()
                return True
            elif action == "lidar_offset_set":
                name = str(payload.get("name", ""))
                target = next((l for l in lidars if l.name == name), None)
                if target is None:
                    return False
                target.set_offset(
                    x=payload.get("x", None),
                    y=payload.get("y", None),
                    z=payload.get("z", None),
                )
                request_config_save()
                return True
            elif action == "lidar_yaw_delta":
                name = str(payload.get("name", ""))
                delta = float(payload.get("delta_deg", 0.0))
                target = next((l for l in lidars if l.name == name), None)
                if target is None:
                    return False
                target.add_yaw_deg(delta)
                request_config_save()
                return True
            elif action == "lidar_yaw_set":
                name = str(payload.get("name", ""))
                yaw_deg = float(payload.get("yaw_deg", 0.0))
                target = next((l for l in lidars if l.name == name), None)
                if target is None:
                    return False
                target.set_yaw_deg(yaw_deg)
                request_config_save()
                return True
            elif action == "lidar_axis_drag":
                name = str(payload.get("name", ""))
                axis = str(payload.get("axis", "")).lower()
                pixels = float(payload.get("pixels", 0.0))
                sensitivity = float(payload.get("sensitivity", 0.05))
                target = next((l for l in lidars if l.name == name), None)
                if target is None:
                    return False
                step_scale = max(0.001, min(1.0, abs(sensitivity)))
                if axis == "yaw":
                    delta = pixels * float(offset_ui_state["yaw_step_deg"]) * step_scale
                    target.add_yaw_deg(delta)
                elif axis == "x":
                    target.add_offset(dx=pixels * float(offset_ui_state["step"]) * step_scale)
                elif axis == "y":
                    target.add_offset(dy=pixels * float(offset_ui_state["step"]) * step_scale)
                elif axis == "z":
                    target.add_offset(dz=pixels * float(offset_ui_state["step"]) * step_scale)
                else:
                    return False
                request_config_save()
                return True
            elif action == "profile_save":
                name = str(payload.get("name", "")).strip()
                if not name:
                    return False
                profiles[name] = {
                    "step": float(offset_ui_state["step"]),
                    "yaw_step_deg": float(offset_ui_state["yaw_step_deg"]),
                    "lidars": {
                        l.name: {
                            "offset": l.get_offset(),
                            "yaw_deg": l.get_yaw_deg(),
                        }
                        for l in lidars
                    },
                }
                ok = persist_profiles()
                if ok:
                    print(f"[Profile] saved: {name}")
                return ok
            elif action == "profile_load":
                name = str(payload.get("name", "")).strip()
                data = profiles.get(name)
                if not isinstance(data, dict):
                    return False
                offset_ui_state["step"] = max(0.001, min(1.0, float(data.get("step", offset_ui_state["step"]))))
                offset_ui_state["yaw_step_deg"] = max(0.1, min(20.0, float(data.get("yaw_step_deg", offset_ui_state["yaw_step_deg"]))))
                lidar_data = data.get("lidars", {})
                for l in lidars:
                    one = lidar_data.get(l.name, {})
                    off = one.get("offset", {})
                    l.set_offset(
                        x=off.get("x", None),
                        y=off.get("y", None),
                        z=off.get("z", None),
                    )
                    if "yaw_deg" in one:
                        l.set_yaw_deg(float(one.get("yaw_deg", 0.0)))
                print(f"[Profile] loaded: {name}")
                return True
            elif action == "profile_delete":
                name = str(payload.get("name", "")).strip()
                if not name or name not in profiles:
                    return False
                del profiles[name]
                ok = persist_profiles()
                if ok:
                    print(f"[Profile] deleted: {name}")
                return ok
            elif action == "auto_icp":
                target_name = str(payload.get("name", "")).strip()
                if not target_name:
                    selected = get_selected_lidar()
                    target_name = selected.name if selected is not None else ""
                
                target = next((l for l in lidars if l.name == target_name), None)
                if target is None:
                    print(f"[Auto ICP] Target {target_name} not found")
                    return False

                if calib_map_points.shape[0] < 64:
                    print(f"[Auto ICP] Not enough ZED map points")
                    return False
                
                pts_local = target.get_latest_points()
                if not pts_local:
                    print(f"[Auto ICP] No lidar points available")
                    return False
                
                best_rmse = float('inf')
                current_off = target.get_offset()
                best_x, best_y, best_z, best_yaw = current_off["x"], current_off["y"], current_off["z"], target.get_yaw_deg()
                
                # Lock initial positions to prevent runaway drift on degenerate environments
                init_x, init_z, init_yaw = best_x, best_z, best_yaw
                max_drift_pos = 0.03  # 최대 3cm (기존 15cm) 이동 제한
                max_drift_yaw = 3.0   # 최대 3도 (기존 30도) 회전 제한
                
                def eval_target(x, y, z, yaw):
                    t_robot_lidar = build_extrinsic_4x4({"x": x, "y": y, "z": z}, yaw)
                    if not pose_history:
                        t_world_robot = np.eye(4, dtype=np.float32)
                    else:
                        t_world_robot, _ = get_pose_nearest_to_time(None, np.eye(4, dtype=np.float32))
                    t_world_lidar = np.dot(t_world_robot, t_robot_lidar)
                    r, n = compute_rmse_points_to_map_local(pts_local, t_world_lidar)
                    return r if r is not None else float('inf')

                current_rmse = eval_target(best_x, best_y, best_z, best_yaw)
                if current_rmse != float('inf'):
                    best_rmse = current_rmse
                
                print(f"[Auto ICP] Starting... initial RMSE: {best_rmse:.5f}")
                step_pos = 0.01  # 탐색 시작 위치 이동 스텝 (기존 5cm -> 1cm)
                step_yaw = 0.5   # 탐색 시작 회전 스텝 (기존 2도 -> 0.5도)
                
                for _ in range(15):
                    improved = False
                    for dx, dy, dz, dyaw in [
                        (step_pos,0,0,0), (-step_pos,0,0,0), 
                        (0,0,step_pos,0), (0,0,-step_pos,0),
                        (0,0,0,step_yaw), (0,0,0,-step_yaw)
                    ]:
                        nx = best_x + dx
                        ny = best_y + dy
                        nz = best_z + dz
                        nyaw = best_yaw + dyaw
                        
                        # Apply bounds to prevent runway sliding along walls
                        if abs(nx - init_x) > max_drift_pos:
                            continue
                        if abs(nz - init_z) > max_drift_pos:
                            continue
                        if abs(nyaw - init_yaw) > max_drift_yaw:
                            continue
                            
                        r = eval_target(nx, ny, nz, nyaw)
                        if r < best_rmse:
                            best_rmse = r
                            best_x, best_y, best_z, best_yaw = nx, ny, nz, nyaw
                            improved = True
                    if not improved:
                        step_pos *= 0.5
                        step_yaw *= 0.5
                        if step_pos < 0.001 and step_yaw < 0.05: # 종료 조건도 더 정밀하게 세팅
                            break
                            
                print(f"[Auto ICP] Done... final RMSE: {best_rmse:.5f}")
                target.set_offset(x=best_x, y=best_y, z=best_z)
                target.set_yaw_deg(best_yaw)
                request_config_save()
                return True
            else:
                return False

            if changed_calib:
                request_config_save()
            off = selected.get_offset()
            yaw = selected.get_yaw_deg()
            print(f"[Offset] {selected.name} -> x={off['x']:+.3f}, y={off['y']:+.3f}, z={off['z']:+.3f}, yaw={yaw:+.2f}deg (step={step:.3f}m/{offset_ui_state['yaw_step_deg']:.2f}deg)")
            return True

        def dispatch_control(action, payload):
            nonlocal save_map_requested, pending_config_save
            if action == "save_spatial_map":
                save_map_requested = True
                return True
            if action == "config_save_now":
                ok = persist_lidar_config()
                if ok:
                    pending_config_save = False
                return ok
            if action == "config_reload_now":
                ok = reload_lidar_config_from_disk()
                if ok:
                    update_control_status()
                return ok
            if action == "calibration_start":
                target_name = str(payload.get("name", "")).strip()
                if not target_name:
                    selected = get_selected_lidar()
                    target_name = selected.name if selected is not None else ""
                calib_state["active"] = True
                calib_state["selected_name"] = target_name
                calib_state["rmse_m"] = None
                calib_state["rmse_ema_m"] = None
                calib_state["sample_count"] = 0
                calib_state["updated_at_s"] = time.time()
                calib_state["note"] = "running"
                rebuild_calib_map_points()
                print(f"[Calib] start target={target_name if target_name else 'selected'} map_pts={calib_state['map_point_count']}")
                return True
            if action == "calibration_stop":
                calib_state["active"] = False
                calib_state["note"] = "stopped"
                calib_state["updated_at_s"] = time.time()
                print("[Calib] stop")
                return True
            if action == "record_start":
                data_logger.start_logging()
                return True
            if action == "record_stop":
                data_logger.stop_logging(zed_map_points=calib_map_points)
                return True
            ok = apply_offset_control(action, payload)
            if ok:
                update_control_status()
            return ok

        update_control_status()
        viewer.set_command_callback(dispatch_control)

        if web_enabled:
            server = web_stream.WebFrameServer(host=web_host, port=web_port)
            def on_web_control(action, payload):
                if viewer is None:
                    return False
                return dispatch_control(action, payload)

            server.set_control_callback(on_web_control)
            server.set_state_callback(get_runtime_lidar_state)
            server.start()
            print(f"[Web] Control panel bind: {web_host}:{web_port}")
            print(f"[Web] Open on this PC: http://localhost:{web_port}/")
        
        print("\n=== Controls ===")
        print("  [Space] : Start/Stop spatial mapping (RGB overlay)")
        print("  [S] : Save current spatial map (mesh_gen.obj)")
        print("  [Mouse Wheel on 3D view] : Zoom in/out")
        print("  [R] : Reset pan (viewer window)")
        print("  [External UI/API] Use /control for x/y/z/yaw/step realtime control")
        print("  [Esc/Q] : Quit")
        
        # Objects
        image = sl.Mat()
        pose = sl.Pose()
        runtime_parameters = sl.RuntimeParameters()
        runtime_ref = _enum_value(
            sl.REFERENCE_FRAME,
            zed_options["runtime"]["measure3D_reference_frame"],
            default_value=None,
            label="zed.runtime.measure3D_reference_frame",
        )
        if runtime_ref is not None:
            runtime_parameters.measure3D_reference_frame = runtime_ref
        print_zed_settings_snapshot("Applied", init, tracking_params, mapping_params, runtime_parameters)
        
        last_call = time.time()
        mapping_activated = False
        should_exit = False

        while viewer.is_available() and not should_exit:
            force_toggle_mapping = False
            force_save_map = False
            if os.name == "nt":
                try:
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        # Ctrl+V in console input stream
                        if key == b"\x16":
                            print("[System] Ctrl+V detected in terminal -> exiting")
                            should_exit = True
                            continue
                        # consume extended key tail byte (arrow/F-keys)
                        if key in (b"\x00", b"\xe0"):
                            if msvcrt.kbhit():
                                msvcrt.getch()
                            continue
                        if key in (b"q", b"Q", b"\x1b"):
                            should_exit = True
                            continue
                        if key == b" ":
                            force_toggle_mapping = True
                        if key in (b"s", b"S"):
                            force_save_map = True
                except Exception:
                    pass
            else:
                key = poll_console_key(console_input_state)
                if key in (b"q", b"Q", b"\x1b"):
                    should_exit = True
                    continue
                if key == b" ":
                    force_toggle_mapping = True
                if key in (b"s", b"S"):
                    force_save_map = True

            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                now_wall = time.time()
                if prev_grab_wall_time is not None:
                    dt = now_wall - float(prev_grab_wall_time)
                    if dt > 1e-6:
                        inst_fps = 1.0 / dt
                        if float(runtime_perf["zed_fps"]) <= 0.0:
                            runtime_perf["zed_fps"] = inst_fps
                        else:
                            runtime_perf["zed_fps"] = (0.85 * float(runtime_perf["zed_fps"])) + (0.15 * inst_fps)
                prev_grab_wall_time = now_wall
                if save_map_requested or force_save_map:
                    save_map_requested = False
                    force_save_map = False
                    if not mapping_activated:
                        print("[ZED] Spatial mapping is not active. Press [Space] first.")
                    else:
                        try:
                            # Match the sample flow: extract whole map, optional mesh filter/texture, then save.
                            zed.extract_whole_spatial_map(pymesh)
                            if draw_mesh_mode and isinstance(pymesh, sl.Mesh):
                                filter_params = sl.MeshFilterParameters()
                                filter_params.set(sl.MESH_FILTER.MEDIUM)
                                pymesh.filter(filter_params, True)
                                viewer.clear_current_mesh()
                                if mapping_params.save_texture:
                                    print(f"Save texture set to : {mapping_params.save_texture}")
                                    pymesh.apply_texture(sl.MESH_TEXTURE_FORMAT.RGBA)
                            filepath = "mesh_gen.obj"
                            status = pymesh.save(filepath)
                            if status:
                                print(f"[ZED] Mesh saved under {filepath}")
                            else:
                                print(f"[ZED] Failed to save mesh under {filepath}")
                            try:
                                zed.disable_spatial_mapping()
                            except Exception:
                                pass
                            mapping_activated = False
                            print("[ZED] Spatial mapping stopped after save.")
                        except Exception as e:
                            print(f"[ZED] Failed to save spatial map: {e}")

                if pending_config_save and (time.time() - last_config_save_time) >= config_save_interval_sec:
                    if persist_lidar_config():
                        pending_config_save = False
                        last_config_save_time = time.time()
                
                # (A) ZED Data
                zed.retrieve_image(image, sl.VIEW.LEFT)
                tracking_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                try:
                    t_world_robot_live = _to_np_4x4(pose.pose_data().m)
                except Exception:
                    t_world_robot_live = np.eye(4, dtype=np.float32)
                pose_history.append((now_wall, np.array(t_world_robot_live, copy=True)))
                update_robot_velocity_from_pose(pose)
                update_control_status()
                try:
                    body_world, heading_world = build_robot_overlay_world_from_pose(
                        pose.pose_data().m,
                        robot_overlay_options,
                    )
                    viewer.set_robot_overlay(
                        body_world,
                        heading_world,
                        enabled=(tracking_state == sl.POSITIONAL_TRACKING_STATE.OK and robot_overlay_options["enabled"]),
                    )
                except Exception:
                    viewer.set_robot_overlay([], [], enabled=False)
                if mapping_activated:
                    mapping_state = zed.get_spatial_mapping_state()
                else:
                    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
                
                # (B) Spatial Mapping Update
                if mapping_activated:
                    duration = time.time() - last_call
                    if duration > 0.2 and viewer.chunks_updated():
                        zed.request_spatial_map_async()
                        last_call = time.time()
                    
                    if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                        zed.retrieve_spatial_map_async(pymesh)
                        viewer.update_chunks()
                        if calib_state["active"]:
                            rebuild_calib_map_points()
                        if last_map_retrieve_time is not None:
                            dt = now_wall - float(last_map_retrieve_time)
                            if dt > 1e-6:
                                inst_hz = 1.0 / dt
                                if float(runtime_perf["map_hz"]) <= 0.0:
                                    runtime_perf["map_hz"] = inst_hz
                                else:
                                    runtime_perf["map_hz"] = (0.85 * float(runtime_perf["map_hz"])) + (0.15 * inst_hz)
                        last_map_retrieve_time = now_wall
                    
                # (C) LiDAR Data Update (multi-lidar)
                lidar_frames = []
                lag_samples_ms = []
                calib_inputs = {}
                for lidar in lidars:
                    pts_local = lidar.get_latest_points()
                    alert_pts_local = lidar.get_latest_alert_points()
                    status = lidar.get_status()
                    frame_time_s = status.get("frame_time_s", None)
                    if frame_time_s is not None:
                        try:
                            frame_time_s = float(frame_time_s)
                        except Exception:
                            frame_time_s = None
                    t_world_robot, lag_ms = get_pose_nearest_to_time(frame_time_s, t_world_robot_live)
                    if lag_ms is not None:
                        lag_samples_ms.append(abs(float(lag_ms)))
                    t_robot_lidar = build_extrinsic_4x4(
                        status.get("offset", {"x": 0.0, "y": 0.0, "z": 0.0}),
                        status.get("yaw_deg", 0.0),
                    )
                    t_world_lidar = np.dot(t_world_robot, t_robot_lidar)
                    calib_inputs[lidar.name] = (pts_local, t_world_lidar)
                    pts_world = transform_points_flat(pts_local, t_world_lidar)
                    alert_pts_world = transform_points_flat(alert_pts_local, t_world_lidar)
                    lidar_frames.append({
                        "name": lidar.name,
                        "points": pts_world,
                        "alert_points": alert_pts_world,
                        "connected": status.get("connected", False),
                        "fps": status.get("fps", 0.0),
                        "offset": status.get("offset", {"x": 0.0, "y": 0.0, "z": 0.0}),
                        "yaw_deg": status.get("yaw_deg", 0.0),
                    })
                    
                    if data_logger.is_logging:
                        target_n = str(calib_state.get("selected_name", "")).strip()
                        if not target_n: target_n = lidars[0].name
                        if lidar.name == target_n:
                            data_logger.log_frame(frame_time_s if frame_time_s else now_wall, t_world_robot, pts_local)
                if lag_samples_ms:
                    lag_avg = float(sum(lag_samples_ms) / max(1, len(lag_samples_ms)))
                    if float(runtime_perf["lidar_pose_lag_ms"]) <= 0.0:
                        runtime_perf["lidar_pose_lag_ms"] = lag_avg
                    else:
                        runtime_perf["lidar_pose_lag_ms"] = (0.8 * float(runtime_perf["lidar_pose_lag_ms"])) + (0.2 * lag_avg)

                if calib_state["active"] and (now_wall - float(calib_last_compute_s)) >= float(calib_compute_interval_s):
                    calib_last_compute_s = now_wall
                    if mapping_state == sl.SPATIAL_MAPPING_STATE.NOT_ENABLED:
                        calib_state["note"] = "mapping_off"
                    elif float(robot_velocity_state.get("speed_mps", 0.0)) > 0.03:
                        calib_state["note"] = "robot_moving"
                    else:
                        target_name = str(calib_state.get("selected_name", "")).strip()
                        if not target_name:
                            selected = get_selected_lidar()
                            target_name = selected.name if selected is not None else ""
                            calib_state["selected_name"] = target_name
                        target_frame = next((f for f in lidar_frames if str(f.get("name", "")) == target_name), None)
                        if target_frame is None:
                            calib_state["note"] = "target_missing"
                        else:
                            if calib_map_points.shape[0] < 64:
                                rebuild_calib_map_points()
                            calib_input = calib_inputs.get(target_name, None)
                            if calib_input is None:
                                rmse, n_used = None, 0
                            else:
                                rmse, n_used = compute_rmse_points_to_map_local(calib_input[0], calib_input[1])
                            calib_state["sample_count"] = int(n_used)
                            calib_state["updated_at_s"] = now_wall
                            if rmse is None:
                                calib_state["note"] = "insufficient_points"
                                calib_state["rmse_m"] = None
                            else:
                                calib_state["rmse_m"] = float(rmse)
                                prev_ema = calib_state.get("rmse_ema_m", None)
                                if prev_ema is None:
                                    calib_state["rmse_ema_m"] = float(rmse)
                                else:
                                    calib_state["rmse_ema_m"] = ((1.0 - calib_rmse_alpha) * float(prev_ema)) + (calib_rmse_alpha * float(rmse))
                                calib_state["note"] = "ok"

                viewer.update_lidar_multi(lidar_frames)
                    
                # (D) Update View
                change_state = viewer.update_view(
                    image,
                    None,
                    pose.pose_data(),
                    tracking_state,
                    mapping_state,
                )
                if change_state or force_toggle_mapping:
                    if not mapping_activated:
                        try:
                            reset_spatial_mapping_session(zed, viewer, pymesh, mapping_params)
                            mapping_activated = True
                            last_call = time.time()
                            print("[ZED] Spatial mapping started.")
                        except Exception as e:
                            print(f"[ZED] Failed to start spatial mapping: {e}")
                    else:
                        try:
                            zed.disable_spatial_mapping()
                            mapping_activated = False
                            print("[ZED] Spatial mapping stopped.")
                        except Exception as e:
                            print(f"[ZED] Failed to stop spatial mapping: {e}")

    except KeyboardInterrupt:
        print("\n[System] KeyboardInterrupt received -> exiting")
    finally:
        restore_console_input(console_input_state)
        print("Exiting...")
        if pending_config_save:
            persist_lidar_config()
        for lidar in lidars:
            lidar.stop()
        for lidar in lidars:
            lidar.join(timeout=2.0)

        if server is not None:
            server.stop()

        if image is not None:
            image.free(memory_type=sl.MEM.CPU)

        if pymesh is not None:
            pymesh.clear()

        if zed is not None:
            try:
                zed.disable_spatial_mapping()
            except Exception:
                pass
            try:
                zed.disable_positional_tracking()
            except Exception:
                pass
            zed.close()

if __name__ == "__main__":
    main()
