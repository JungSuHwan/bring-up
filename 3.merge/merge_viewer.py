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
    parser = argparse.ArgumentParser(description="ZED + LiDAR merge viewer")
    parser.add_argument("--lidar-config", default="lidar_config.json", help="LiDAR config path (default: lidar_config.json)")
    parser.add_argument("--web", action="store_true", help="Enable web streaming")
    parser.add_argument("--web-host", default="0.0.0.0", help="Web server host (default: 0.0.0.0)")
    parser.add_argument("--web-port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--web-fps", type=int, default=5, help="Web stream capture FPS (default: 5)")
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
        "fps": int(web_cfg.get("fps", 5)),
    }


def load_lidar_receivers(config_path, config=None):
    config_abs_path = os.path.abspath(config_path)
    print(f"[LiDAR] Using config: {config_abs_path}")

    # Backward-compatible default: single LiDAR if config is absent.
    if config is None:
        config = load_config_json(config_path)

    lidar_items = config.get("lidars", [])
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

        receiver = lidar_thread.LidarReceiver(
            ip=ip,
            port=port,
            name=name,
            offset_x=offset_x,
            offset_y=offset_y,
            offset_z=offset_z,
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
    pc_window_enabled = display_options["pc_window_enabled"]
    # CLI flag still works. If not given, config can enable web streaming.
    web_enabled = bool(args.web or web_options["enabled"])
    web_host = args.web_host if args.web else web_options["host"]
    web_port = args.web_port if args.web else web_options["port"]
    web_fps = args.web_fps if args.web else web_options["fps"]

    try:
        # 1. Initialize ZED
        print("Initializing ZED Camera...")
        init = sl.InitParameters()
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init.depth_maximum_distance = 10.0
        
        zed = sl.Camera()
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Camera Open Failed: {status}")
            return

        # 2. Enable Tracking & Mapping
        print("Enabling Tracking and Spatial Mapping...")
        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.enable_area_memory = False
        
        # Set initial position (Camera at 30cm height)
        initial_position = sl.Transform()
        initial_translation = sl.Translation()
        initial_translation.init_vector(0, 0.30, 0) # x, y, z
        initial_position.set_translation(initial_translation)
        tracking_params.set_initial_world_transform(initial_position)
        
        tracking_err = zed.enable_positional_tracking(tracking_params)
        if tracking_err != sl.ERROR_CODE.SUCCESS:
            print(f"Enable positional tracking failed: {tracking_err}")
            return
        
        mapping_params = sl.SpatialMappingParameters()
        mapping_params.map_type = sl.SPATIAL_MAP_TYPE.MESH
        mapping_params.save_texture = True
        mapping_params.resolution_meter = mapping_params.get_resolution_preset(sl.MAPPING_RESOLUTION.MEDIUM)
        mapping_params.range_meter = mapping_params.get_range_preset(sl.MAPPING_RANGE.MEDIUM)
        mapping_params.use_chunk_only = True
        
        # 3. Start LiDAR Threads
        lidars = load_lidar_receivers(args.lidar_config, config=config)
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
        print(f"[Display] PC window enabled: {pc_window_enabled}")

        reset_spatial_mapping_session(zed, viewer, pymesh, mapping_params)
        print("[ZED] Spatial mapping session reset complete (fresh start).")

        if web_enabled:
            server = web_stream.WebFrameServer(host=web_host, port=web_port)
            server.start()
            viewer.set_frame_callback(server.update_frame, fps=web_fps)
            print(f"[Web] Stream bind: {web_host}:{web_port} (fps={web_fps})")
            print(f"[Web] Open on this PC: http://localhost:{web_port}/")
        
        print("\n=== Controls ===")
        print("  [Space] : Pause/Resume Spatial Mapping")
        print("  [Esc/Q] : Quit")
        
        # Objects
        image = sl.Mat()
        pose = sl.Pose()
        runtime_parameters = sl.RuntimeParameters()
        
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
                        # consume extended key tail byte
                        if key in (b"\x00", b"\xe0") and msvcrt.kbhit():
                            msvcrt.getch()
                except Exception:
                    pass

            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                
                # (A) ZED Data
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
                    status = lidar.get_status()
                    lidar_frames.append({
                        "name": lidar.name,
                        "points": pts if pts else [],
                        "connected": status.get("connected", False),
                        "fps": status.get("fps", 0.0),
                    })

                viewer.update_lidar_multi(lidar_frames)
                    
                # (D) Update View
                viewer.update_view(image, None, pose.pose_data(), tracking_state, mapping_state)

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
