import sys
import time
import signal
import numpy as np
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import lidar_thread

def main():
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
    
    # Set initial position (Camera at 30cm height)
    initial_position = sl.Transform()
    initial_translation = sl.Translation()
    initial_translation.init_vector(0, 0.30, 0) # x, y, z
    initial_position.set_translation(initial_translation)
    tracking_params.set_initial_world_transform(initial_position)
    
    zed.enable_positional_tracking(tracking_params)
    
    mapping_params = sl.SpatialMappingParameters()
    mapping_params.map_type = sl.SPATIAL_MAP_TYPE.MESH
    mapping_params.save_texture = True
    mapping_params.resolution_meter = mapping_params.get_resolution_preset(sl.MAPPING_RESOLUTION.MEDIUM)
    mapping_params.range_meter = mapping_params.get_range_preset(sl.MAPPING_RANGE.MEDIUM)
    mapping_params.use_chunk_only = True
    zed.enable_spatial_mapping(mapping_params)
    
    # 3. Start LiDAR Thread
    print("Starting LiDAR Receiver...")
    lidar = lidar_thread.LidarReceiver() # Default IP/Port
    lidar.start()

    # 4. Initialize Viewer
    camera_info = zed.get_camera_information()
    viewer = gl.GLViewer()
    pymesh = sl.Mesh()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, pymesh, True)
    
    print("\n=== Controls ===")
    print("  [Space] : Pause/Resume Spatial Mapping")
    print("  [Esc/Q] : Quit")
    
    # Objects
    image = sl.Mat()
    pose = sl.Pose()
    runtime_parameters = sl.RuntimeParameters()
    
    last_call = time.time()
    
    while viewer.is_available():
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
                
            # (C) LiDAR Data Update
            lidar_points = lidar.get_latest_points()
            if lidar_points:
                viewer.update_lidar(lidar_points)
                
            # (D) Update View
            # Note: We are not visualizing depth image in this merged view for simplicity, or we can add it if needed.
            # viewer.update_view expects depth pointer?
            # Let's see viewer.py signature: update_view(self, _image, _depth_ptr, _pose, ...)
            # If we don't pass depth, logic handles it?
            # "if _depth_ptr is not None: update_texture..."
            # So passing None is safe.
            
            viewer.update_view(image, None, pose.pose_data(), tracking_state, mapping_state)
            
    # Cleanup
    print("Exiting...")
    lidar.stop()
    lidar.join()
    
    image.free(memory_type=sl.MEM.CPU)
    pymesh.clear()
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()

if __name__ == "__main__":
    main()
