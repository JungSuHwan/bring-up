
import sys
import ctypes
import time
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv2
import numpy as np



def main():
    # 1. ZED 초기화 (Initialize ZED)
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL 
    init.coordinate_units = sl.UNIT.METER  
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP 
    init.depth_maximum_distance = 8.0     
    
    # 2. 카메라 열기 (Open Camera)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open Failed: {status}")
        exit()
    
    print("ZED Camera Connected.")

    # 3. 위치 추적 활성화
    tracking_params = sl.PositionalTrackingParameters()
    status = zed.enable_positional_tracking(tracking_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Enable Tracking Failed: {status}")
        exit()

    # 4. 공간 매핑 설정
    mapping_params = sl.SpatialMappingParameters()
    mapping_params.map_type = sl.SPATIAL_MAP_TYPE.MESH
    mapping_params.save_texture = True 
    mapping_params.resolution_meter = mapping_params.get_resolution_preset(sl.MAPPING_RESOLUTION.MEDIUM)
    mapping_params.range_meter = mapping_params.get_range_preset(sl.MAPPING_RANGE.MEDIUM)
    mapping_params.use_chunk_only = True

    # 5. 공간 매핑 즉시 활성화
    zed.enable_spatial_mapping(mapping_params)
    mapping_activated = True
    print("Spatial Mapping STARTED (Auto)")

    # 필요한 객체 생성
    image = sl.Mat()
    depth = sl.Mat()
    pose = sl.Pose()
    pymesh = sl.Mesh()
    
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    
    # 6. OpenGL 뷰어 초기화
    viewer = gl.GLViewer()
    viewer.init(zed.get_camera_information().camera_configuration.calibration_parameters.left_cam, pymesh, True)
    
    print("\n=== Controls ===")
    print("  [Space]   : Pause/Resume Spatial Mapping")
    print("  [Esc/Q]   : Quit")
    print("================\n")
    
    last_call = time.time()
    last_fps_time = time.time()
    frame_count = 0
    fps = 0.0
    
    # 메인 루프
    while viewer.is_available():
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            
            # (1) 데이터 수집
            zed.retrieve_image(image, sl.VIEW.LEFT)     # RGB
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH) # Depth
            tracking_state = zed.get_position(pose)     # Pose
            
            # (2) FPS 계산
            frame_count += 1
            if frame_count >= 30:
                current_time = time.time()
                fps = frame_count / (current_time - last_fps_time)
                last_fps_time = current_time
                frame_count = 0
                print(f"FPS: {fps:.2f}")

            # (3) Depth 시각화 데이터 준비 (Colorized Depth)
            depth_np = depth.get_data()
            depth_display = np.copy(depth_np)
            depth_display = np.nan_to_num(depth_display, nan=0.0, posinf=0.0, neginf=0.0)
            depth_display = (depth_display / 5.0 * 255.0).clip(0, 255).astype(np.uint8)
            # Apply Colormap (JET) using OpenCV
            # Result is BGR (H, W, 3). Add Alpha for GL (H, W, 4)
            depth_colormap_bgr = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            depth_colormap_bgra = cv2.cvtColor(depth_colormap_bgr, cv2.COLOR_BGR2BGRA)
            
            # Ensure data is contiguous for ctypes
            depth_colormap_bgra = np.ascontiguousarray(depth_colormap_bgra)
            
            # Get Pointer
            depth_ptr = ctypes.c_void_p(depth_colormap_bgra.ctypes.data)

            # (4) Spatial Mapping 업데이트 요청
            mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
            if mapping_activated:
                mapping_state = zed.get_spatial_mapping_state()
                duration = time.time() - last_call
                if duration > 0.2 and viewer.chunks_updated():
                    zed.request_spatial_map_async()
                    last_call = time.time()
                
                if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_spatial_map_async(pymesh)
                    viewer.update_chunks()

            # (5) OpenGL 뷰어 업데이트 (RGB Image + Depth Pointer)
            viewer.update_view(image, depth_ptr, pose.pose_data(), tracking_state, mapping_state)

    # 종료
    image.free(memory_type=sl.MEM.CPU)
    depth.free(memory_type=sl.MEM.CPU)
    pymesh.clear()
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    print("Program Finished.")

if __name__ == "__main__":
    main()
