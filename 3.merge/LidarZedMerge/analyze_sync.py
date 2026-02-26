import os
import sys
import glob
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
    key_rots = Rotation.from_matrix(np.stack([p0[:3, :3], p1[:3, :3]]))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    interp_rot = slerp([alpha])[0].as_matrix()
    
    interp_pose = np.eye(4)
    interp_pose[:3, :3] = interp_rot
    interp_pose[:3, 3] = trans
    
    return interp_pose

def analyze(session_dir):
    print(f"Loading data from {session_dir} ...")
    timestamps, poses, zed_map, lidar_files = load_data(session_dir)
    
    if timestamps is None:
        sys.exit(1)

    print(f"Loaded {len(timestamps)} poses, {len(zed_map)} map points, {len(lidar_files)} LiDAR sweeps.")

    print("Building KDTree for ZED Map (This takes a few seconds)...")
    tree = cKDTree(zed_map)
    
    # 분석할 오차 시간(옵셋) 범위 설정: -100ms ~ +100ms 를 5ms 간격으로 테스트
    offsets_ms = np.arange(-100, 105, 5) 
    offsets_s = offsets_ms / 1000.0
    
    rmse_list = []
    
    # 라이다 데이터를 한 번에 메모리에 로드 (속도 최적화)
    lidar_data = []
    for lf in lidar_files:
        pts = np.load(lf).reshape(-1, 3) # (N, 3) 형태로 변환
        # 파일 이름에서 타임스탬프 추출 ("1705648834.123456.npy" -> 1705648834.123456)
        file_ts = float(os.path.basename(lf).replace(".npy", ""))
        lidar_data.append((file_ts, pts))
    
    print("\n=== Testing time offsets... ===\n")
    for offset in offsets_s:
        total_squared_error = 0.0
        total_points = 0
        
        for file_ts, pts in lidar_data:
            # 핵심! 라이다 도착시간에 offset을 감하여 "과거 진짜 출발시간"을 찾습니다.
            eval_time = file_ts + offset 
            
            # 해당 진짜 출발시간에 로봇이 어디 있었는지 위치 보간
            pose = interpolate_pose(eval_time, timestamps, poses)
            
            # 라이다 점들을 로봇 위치 기준으로 월드 좌표(World Frame)로 변환
            pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
            pts_w = (pose @ pts_h.T).T[:, :3]
            
            # ZED 맵과 가장 가까운 거리(오차) 계산
            dists, _ = tree.query(pts_w, k=1)
            
            # 바닥이나 허공의 노이즈 점들을 무시하기 위해 유효 범위(1.5m 이내) 필터링
            valid = dists[dists < 1.5]
            if len(valid) > 0:
                # 이상치(Outlier)를 한번 더 걸러내기 위해 하위 90%만 사용
                p90 = np.percentile(valid, 90)
                inliers = valid[valid <= p90]
                total_squared_error += np.sum(inliers**2)
                total_points += len(inliers)
                
        if total_points > 0:
            rmse = np.sqrt(total_squared_error / total_points)
        else:
            rmse = float('inf')
            
        rmse_list.append(rmse)
        print(f"Offset {offset*1000:5.0f} ms -> RMSE: {rmse:.4f} m")
        
    best_idx = np.argmin(rmse_list)
    best_offset_ms = offsets_ms[best_idx]
    best_rmse = rmse_list[best_idx]
    
    print("\n" + "="*50)
    print(f"BEST TIME SYNC OFFSET: {best_offset_ms:+} ms")
    print(f"   (Minimum RMSE: {best_rmse:.4f} m)")
    print("="*50)
    
    # 꺾은선 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(offsets_ms, rmse_list, marker='o', linewidth=2, color='b')
    plt.axvline(best_offset_ms, color='r', linestyle='--', label=f'Best Offset: {best_offset_ms}ms')
    
    # 가장 낮은 점 강조
    plt.plot(best_offset_ms, best_rmse, 'ro', markersize=10)
    plt.text(best_offset_ms, best_rmse + 0.005, f"{best_offset_ms}ms ({best_rmse:.3f}m)", color='red', ha='center')

    plt.title('Time Sync Offset vs RMSE', fontsize=16)
    plt.xlabel('Time Offset (ms)', fontsize=14)
    plt.ylabel('RMSE (m)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    # 결과를 파일로 저장
    save_path = os.path.join(session_dir, "sync_result.png")
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
