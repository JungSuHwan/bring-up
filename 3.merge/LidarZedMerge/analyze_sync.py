import os
import sys
import glob
import json
import math
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\n[Error] 'matplotlib' 라이브러리가 설치되어 있지 않습니다!")
    print("그래프를 그리기 위해 필수입니다. 설치 명령어: pip install matplotlib\n")
    sys.exit(1)
from scipy.spatial.transform import Rotation, Slerp

def load_data(session_dir):
    traj_path = os.path.join(session_dir, "trajectory.npz")
    map_path = os.path.join(session_dir, "zed_map.npy")
    lidar_dir = os.path.join(session_dir, "lidar_points")

    if not os.path.exists(traj_path) or not os.path.exists(map_path) or not os.path.exists(lidar_dir):
        print(f"[Error] Data files missing in {session_dir}")
        print("Make sure 'trajectory.npz', 'zed_map.npy', and 'lidar_points' folder exist.")
        return None, None, None, None

    try:
        traj_data = np.load(traj_path)
        timestamps = traj_data['timestamps']
        poses = traj_data['poses']
        zed_map = np.load(map_path)
    except Exception as e:
        print(f"[Error] Failed to load Numpy arrays: {e}")
        return None, None, None, None
    
    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.npy")))
    if not lidar_files:
        print(f"[Error] No LiDAR points found in {lidar_dir}")
        return None, None, None, None
        
    return timestamps, poses, zed_map, lidar_files

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

def load_t_robot_lidar():
    # lidar_config.json에서 LiDAR 회전 및 오프셋 보정값을 읽어옵니다.
    try:
        with open("lidar_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
            lidar_cfg = cfg.get("lidars", [{}])[0]
            offset = lidar_cfg.get("offset", {"x": 0.0, "y": 0.0, "z": 0.0})
            yaw = lidar_cfg.get("rotation", {}).get("yaw_deg", 0.0)
            return build_extrinsic_4x4(offset, yaw)
    except Exception as e:
        print(f"[Warning] Failed to load extrinsic: {e}")
        return np.eye(4, dtype=np.float32)

def interpolate_pose(eval_time, timestamps, poses):
    """
    주어진 시간(eval_time)에 해당하는 로봇의 3D 포즈(4x4 행렬)를 
    과거 기록(timestamps)을 바탕으로 앞뒤를 비례 배분(Slerp/Lerp)하여 찾습니다.
    """
    if eval_time <= timestamps[0]:
        return poses[0]
    if eval_time >= timestamps[-1]:
        return poses[-1]
    
    idx = np.searchsorted(timestamps, eval_time)
    t0, t1 = timestamps[idx - 1], timestamps[idx]
    dt = t1 - t0
    
    if dt < 1e-6:
        return poses[idx]
    
    alpha = (eval_time - t0) / dt
    
    p0 = poses[idx-1]
    p1 = poses[idx]
    
    # 1. Translation 보간 (Lerp)
    trans0 = p0[:3, 3]
    trans1 = p1[:3, 3]
    trans = trans0 + alpha * (trans1 - trans0)
    
    # 2. Rotation 보간 (Slerp)
    try:
        key_rots = Rotation.from_matrix(np.stack([p0[:3, :3], p1[:3, :3]]))
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp([alpha])[0].as_matrix()
    except Exception:
        interp_rot = p0[:3, :3]
    
    interp_pose = np.eye(4)
    interp_pose[:3, :3] = interp_rot
    interp_pose[:3, 3] = trans
    
    return interp_pose

def deskew_points_offline(pts, file_ts, offset, timestamps, poses, t_robot_lidar, scan_duration=0.1):
    """
    저장된 로컬 pts(N, 3)는 시간에 따라 순차적으로 파싱된 순서입니다.
    이를 바탕으로 인덱스에 비례해 스캔 시간을 분배하여 보간(Deskewing)합니다.
    (기존 각도 역산 방식은 -pi ~ pi wrap-around 시 시간 배분이 꼬이는 버그 방지)
    """
    n = len(pts)
    if n == 0:
        return np.zeros_like(pts)
        
    alphas = np.linspace(0.0, 1.0, n)
    scan_t1 = file_ts + offset
    scan_t0 = scan_t1 - scan_duration
    
    eval_times = scan_t0 + alphas * scan_duration
    
    world_pts = np.zeros_like(pts)
    pts_h = np.concatenate([pts, np.ones((n, 1))], axis=1)
    
    for i in range(n):
        pose = interpolate_pose(eval_times[i], timestamps, poses)
        pose_lidar = pose @ t_robot_lidar
        world_pts[i] = (pose_lidar @ pts_h[i])[:3]
        
    return world_pts

def analyze(session_dir):
    print(f"Loading data from {session_dir} ...")
    timestamps, poses, zed_map, lidar_files = load_data(session_dir)
    
    if timestamps is None:
        sys.exit(1)

    print(f"Loaded {len(timestamps)} poses, {len(zed_map)} map points, {len(lidar_files)} LiDAR sweeps.")

    print("Building KDTree for ZED Map (This takes a few seconds)...")
    tree = cKDTree(zed_map)
    t_robot_lidar = load_t_robot_lidar()
    
    # 분석할 오차 시간(옵셋) 범위 설정: -200ms ~ +100ms 를 10ms 간격으로 넓게 테스트
    offsets_ms = np.arange(-200, 105, 10) 
    offsets_s = offsets_ms / 1000.0
    
    rmse_rigid_list = []
    rmse_deskew_list = []
    
    # 라이다 데이터를 한 번에 메모리에 로드 (속도 최적화)
    lidar_data = []
    for lf in lidar_files:
        pts = np.load(lf).reshape(-1, 3) # (N, 3) 형태로 변환
        file_ts = float(os.path.basename(lf).replace(".npy", ""))
        lidar_data.append((file_ts, pts))
    
    print("\n=== Testing time offsets (Rigid vs Deskew) ===\n")
    for offset in offsets_s:
        total_sq_err_rigid = 0.0
        total_sq_err_deskew = 0.0
        pts_count_rigid = 0
        pts_count_deskew = 0
        
        for file_ts, pts in lidar_data:
            eval_time = file_ts + offset 
            
            # [방식 1] 기존: 전체 점을 스캔 종료 시점 하나의 포즈로 변환 (Rigid)
            pose_rigid = interpolate_pose(eval_time, timestamps, poses)
            pose_lidar_rigid = pose_rigid @ t_robot_lidar
            pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
            pts_w_rigid = (pose_lidar_rigid @ pts_h.T).T[:, :3]
            
            dists_rigid, _ = tree.query(pts_w_rigid, k=1)
            valid_r = dists_rigid[dists_rigid < 1.0]
            if len(valid_r) > 0:
                p90_r = np.percentile(valid_r, 90)
                inliers_r = valid_r[valid_r <= p90_r]
                total_sq_err_rigid += np.sum(inliers_r**2)
                pts_count_rigid += len(inliers_r)
                
            # [방식 2] 3단계: 포인트별 시점에 따른 보간 (Deskew)
            pts_w_deskew = deskew_points_offline(pts, file_ts, offset, timestamps, poses, t_robot_lidar)
            dists_deskew, _ = tree.query(pts_w_deskew, k=1)
            valid_d = dists_deskew[dists_deskew < 1.0]
            if len(valid_d) > 0:
                p90_d = np.percentile(valid_d, 90)
                inliers_d = valid_d[valid_d <= p90_d]
                total_sq_err_deskew += np.sum(inliers_d**2)
                pts_count_deskew += len(inliers_d)
                
        rmse_r = np.sqrt(total_sq_err_rigid / pts_count_rigid) if pts_count_rigid > 0 else float('inf')
        rmse_d = np.sqrt(total_sq_err_deskew / pts_count_deskew) if pts_count_deskew > 0 else float('inf')
            
        rmse_rigid_list.append(rmse_r)
        rmse_deskew_list.append(rmse_d)
        print(f"Offset {offset*1000:5.0f} ms -> RMSE Rigid: {rmse_r:.4f} m | Deskew: {rmse_d:.4f} m")
        
    best_idx_r = np.argmin(rmse_rigid_list)
    best_idx_d = np.argmin(rmse_deskew_list)
    
    print("\n" + "="*50)
    print(f"BEST RESULT (Rigid)  : {offsets_ms[best_idx_r]:+} ms | RMSE: {rmse_rigid_list[best_idx_r]:.4f} m")
    print(f"BEST RESULT (Deskew) : {offsets_ms[best_idx_d]:+} ms | RMSE: {rmse_deskew_list[best_idx_d]:.4f} m")
    print(f"    => Deskewing Improved RMSE by: {rmse_rigid_list[best_idx_r] - rmse_deskew_list[best_idx_d]:.5f} m")
    print("="*50)
    
    # 꺾은선 그래프 그리기
    plt.figure(figsize=(10, 6))
    
    # 기존 Rigid 라인
    plt.plot(offsets_ms, rmse_rigid_list, marker='o', linewidth=2, color='gray', label='Without Deskew (Rigid)', alpha=0.6)
    # Deskew 라인
    plt.plot(offsets_ms, rmse_deskew_list, marker='s', linewidth=2, color='b', label='With Deskew')
    
    best_r_ms, best_r_val = offsets_ms[best_idx_r], rmse_rigid_list[best_idx_r]
    best_d_ms, best_d_val = offsets_ms[best_idx_d], rmse_deskew_list[best_idx_d]
    
    # 가장 낮은 점 강조
    plt.plot(best_r_ms, best_r_val, 'ro', markersize=8)
    plt.plot(best_d_ms, best_d_val, 'ro', markersize=10)
    plt.text(best_r_ms, best_r_val + 0.003, f"{best_r_val:.3f}m", color='gray', ha='center')
    plt.text(best_d_ms, best_d_val - 0.005, f"Best: {best_d_val:.4f}m", color='blue', ha='center')

    plt.title('Time Sync & Motion Deskewing RMSE Analysis', fontsize=16)
    plt.xlabel('Time Offset (ms)', fontsize=14)
    plt.ylabel('RMSE (m) [Lower is better]', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    # 결과를 파일로 저장
    save_path = os.path.join(session_dir, "deskew_sync_result.png")
    plt.savefig(save_path, dpi=150)
    print(f"Graph saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_sync.py <session_dir_path>")
        print("Example: python analyze_sync.py logs/session_20260226_155016")
        sys.exit(1)
        
    session_path = sys.argv[1]
    analyze(session_path)
