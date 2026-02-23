import sys
import time
import signal
import argparse
import json
import os
import numpy as np
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import lidar_thread
import web_stream

if os.name == "nt":
    import msvcrt

def parse_args():
    default_lidar_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lidar_config.json")
    parser = argparse.ArgumentParser(description="ZED + LiDAR merge viewer")
    parser.add_argument(
        "--lidar-config",
        default=default_lidar_config,
        help="LiDAR config path (default: LidarZedMerge/lidar_config.json)",
    )
    parser.add_argument("--web", action="store_true", help="Enable web streaming")
    parser.add_argument("--web-host", default="0.0.0.0", help="Web server host (default: 0.0.0.0)")
    parser.add_argument("--web-port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--web-fps", type=int, default=60, help="Web stream capture FPS (default: 60)")
    return parser.parse_args()


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
        "fps": int(web_cfg.get("fps", 60)),
        "optimize_for_web_only": bool(web_cfg.get("optimize_for_web_only", True)),
        "jpeg_quality": int(web_cfg.get("jpeg_quality", 70)),
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
    args = parse_args()
    server = None
    viewer = None
    zed = None
    image = None
    pymesh = None
    lidars = []
    config = load_config_json(args.lidar_config)
    display_options = load_display_options(config)
    web_options = load_web_options(config)
    zed_options = load_zed_options(config)
    pc_window_enabled = display_options["pc_window_enabled"]
    # CLI flag still works. If not given, config can enable web streaming.
    web_enabled = bool(args.web or web_options["enabled"])
    web_host = args.web_host if args.web else web_options["host"]
    web_port = args.web_port if args.web else web_options["port"]
    web_fps = args.web_fps if args.web else web_options["fps"]
    web_optimize_for_web_only = bool(web_options["optimize_for_web_only"])
    web_jpeg_quality = int(web_options["jpeg_quality"])
    config_path = os.path.abspath(args.lidar_config)
    profiles_path = os.path.abspath("lidar_profiles.json")
    offset_ui_state = {
        "selected_idx": 0,
        "step": 0.01,  # meter
        "yaw_step_deg": 0.5,  # degree
    }
    alert_ui_state = load_lidar_alert_options(config)
    profiles = {}
    pending_config_save = False
    last_config_save_time = 0.0
    config_save_interval_sec = 0.5

    try:
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
        lidars = load_lidar_receivers(args.lidar_config, config=config)
        try:
            if os.path.exists(profiles_path):
                with open(profiles_path, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                    if not isinstance(profiles, dict):
                        profiles = {}
        except Exception as e:
            print(f"[Profile] Failed to load {profiles_path}: {e}")
            profiles = {}
        print(f"Starting LiDAR Receivers... count={len(lidars)}")
        for lidar in lidars:
            lidar.start()

        # 4. Initialize Viewer
        camera_info = zed.get_camera_information()
        viewer = gl.GLViewer()
        pymesh = sl.Mesh()
        viewer.init(
            camera_info.camera_configuration.calibration_parameters.left_cam,
            pymesh,
            True,
            show_window=pc_window_enabled,
        )
        web_render_optimized = bool(web_enabled and (not pc_window_enabled) and web_optimize_for_web_only)
        if web_render_optimized:
            viewer.set_stream_3d_only(True)
        print(f"[Display] PC window enabled: {pc_window_enabled}")
        if web_render_optimized:
            print("[Display] Web-only optimization enabled: 3D-only render path")

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
                "profiles": sorted([str(k) for k in profiles.keys()]),
                "lidars": items,
            }

        def update_control_status():
            selected = get_selected_lidar()
            if selected is None:
                if viewer is not None:
                    viewer.set_control_status("LiDAR: none")
                return
            status = selected.get_status()
            off = status.get("offset", {"x": 0.0, "y": 0.0, "z": 0.0})
            yaw = float(status.get("yaw_deg", 0.0))
            text = (
                f"lidar={selected.name} "
                f"step={offset_ui_state['step']:.3f}m yaw_step={offset_ui_state['yaw_step_deg']:.2f}deg "
                f"off=({float(off.get('x', 0.0)):+.3f},{float(off.get('y', 0.0)):+.3f},{float(off.get('z', 0.0)):+.3f}) "
                f"yaw={yaw:+.2f}"
            )
            if viewer is not None:
                viewer.set_control_status(text)

        def persist_profiles():
            try:
                with open(profiles_path, "w", encoding="utf-8") as f:
                    json.dump(profiles, f, indent=2)
                return True
            except Exception as e:
                print(f"[Profile] Failed to save {profiles_path}: {e}")
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
                return True
            except Exception as e:
                print(f"[Config] Failed to save lidar config {config_path}: {e}")
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
            else:
                return False

            if changed_calib:
                request_config_save()
            off = selected.get_offset()
            yaw = selected.get_yaw_deg()
            print(f"[Offset] {selected.name} -> x={off['x']:+.3f}, y={off['y']:+.3f}, z={off['z']:+.3f}, yaw={yaw:+.2f}deg (step={step:.3f}m/{offset_ui_state['yaw_step_deg']:.2f}deg)")
            return True

        def dispatch_control(action, payload):
            ok = apply_offset_control(action, payload)
            if ok:
                update_control_status()
            return ok

        update_control_status()

        reset_spatial_mapping_session(zed, viewer, pymesh, mapping_params)
        print("[ZED] Spatial mapping session reset complete (fresh start).")

        if web_enabled:
            server = web_stream.WebFrameServer(host=web_host, port=web_port)
            def on_web_control(action, payload):
                if viewer is None:
                    return False
                if action == "pan_pixels":
                    dx = float(payload.get("dx", 0.0))
                    dy = float(payload.get("dy", 0.0))
                    viewer.pan_by_pixels(dx, dy)
                    return True
                if action == "zoom_steps":
                    steps = float(payload.get("steps", 0.0))
                    viewer.zoom_by_steps(steps)
                    return True
                if action == "reset_view":
                    viewer.reset_pan_zoom()
                    return True
                return dispatch_control(action, payload)

            server.set_control_callback(on_web_control)
            server.set_state_callback(get_runtime_lidar_state)
            server.set_jpeg_quality(web_jpeg_quality)
            server.start()
            viewer.set_frame_callback(server.update_frame, fps=web_fps)
            print(f"[Web] Stream bind: {web_host}:{web_port} (fps={web_fps})")
            print(f"[Web] Open on this PC: http://localhost:{web_port}/")
        
        print("\n=== Controls ===")
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
        should_exit = False

        while viewer.is_available() and not should_exit:
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
                except Exception:
                    pass

            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                if pending_config_save and (time.time() - last_config_save_time) >= config_save_interval_sec:
                    if persist_lidar_config():
                        pending_config_save = False
                        last_config_save_time = time.time()
                
                # (A) ZED Data
                if not web_render_optimized:
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                tracking_state = zed.get_position(pose)
                mapping_state = zed.get_spatial_mapping_state()
                
                # (B) Spatial Mapping Update
                duration = time.time() - last_call
                if duration > 0.2 and viewer.chunks_updated():
                    zed.request_spatial_map_async()
                    last_call = time.time()
                    
                if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_spatial_map_async(pymesh)
                    viewer.update_chunks()
                    
                # (C) LiDAR Data Update (multi-lidar)
                lidar_frames = []
                for lidar in lidars:
                    pts = lidar.get_latest_points()
                    alert_pts = lidar.get_latest_alert_points()
                    status = lidar.get_status()
                    lidar_frames.append({
                        "name": lidar.name,
                        "points": pts if pts else [],
                        "alert_points": alert_pts if alert_pts else [],
                        "connected": status.get("connected", False),
                        "fps": status.get("fps", 0.0),
                        "offset": status.get("offset", {"x": 0.0, "y": 0.0, "z": 0.0}),
                        "yaw_deg": status.get("yaw_deg", 0.0),
                    })

                viewer.update_lidar_multi(lidar_frames)
                    
                # (D) Update View
                viewer.update_view(image if not web_render_optimized else None, None, pose.pose_data(), tracking_state, mapping_state)

                # When PC window is hidden, force one render pass so web stream receives
                # the merged OpenGL frame (RGB + mesh + lidar) instead of raw camera image.
                if web_enabled and (not pc_window_enabled):
                    try:
                        viewer.draw_callback()
                    except Exception:
                        pass
    except KeyboardInterrupt:
        print("\n[System] KeyboardInterrupt received -> exiting")
    finally:
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
