// ----------------------------------------------------------------------------
// ZED Multi-Camera Headless Client (Multi-Marker & Drift Correction)
// ----------------------------------------------------------------------------

#ifdef _DEBUG
#error Please build the project in Release mode
#endif

// Standard Includes
#include <stdio.h>
#include <string.h>
#include <vector>
#include <memory>
#include <cstring>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <future>
#include <mutex>
#include <thread>
#include <map>
#include <atomic>
#include <iostream>

// ZED SDK
#include <sl/Camera.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

// User Headers
#include "yolo.hpp" 
#include "aruco.hpp"
#include <sio_client.h>

// Namespaces
using namespace std;
using namespace sl;
using namespace nvinfer1;

// ----------------------------------------------------------------------------
// GLOBAL VARIABLES & CONSTANTS
// ----------------------------------------------------------------------------
#define CONF_THRESH 0.3

sio::client socket_client;

// Flags
bool quit = false;
bool record_status = false;
bool reset_status = false;
bool object_detect_status = false;
bool pose_update = false;

std::atomic<bool> is_relocated(false);
float marker_size = 0.16f; // 마커 실제 크기 (미터)

// [NEW] 설정 파일에서 로드할 변수들
float drift_threshold = 0.05f; // 기본값 5cm (파일에서 덮어씀)
std::map<int, sl::Transform> known_markers_map; // ID -> World Transform

std::mutex detector_mutex;

sl::Transform tf[2] = {
    sl::Transform(sl::Rotation(), sl::Translation(0, 0, 0.5)),
    sl::Transform(sl::Rotation(M_PI, sl::Translation(0, 1, 0)), sl::Translation(0, 0, 0))
};

// ----------------------------------------------------------------------------
// DATA STRUCTURES
// ----------------------------------------------------------------------------
struct CameraData {
    Camera zed;
    Mat point_cloud;
    Mat im_left;
    cv::Matx33d camera_matrix = cv::Matx33d::eye();
    Pose pose;
    POSITIONAL_TRACKING_STATE tracking_state = POSITIONAL_TRACKING_STATE::OFF;

    std::chrono::steady_clock::time_point kLastPoseSend = std::chrono::steady_clock::now();
    chrono::high_resolution_clock::time_point ts_last;
    chrono::high_resolution_clock::time_point obj_last = chrono::high_resolution_clock::now();

    bool request_new_mesh = true;
    bool wait_for_mapping = true;
    bool first_camera = false;
    int id = -1;
    bool is_record = false;
    string info;

    Mesh map;
    uint32_t mapping_index = 0;
    sl::Objects objects;
    sl::Transform tf;
    CUstream zed_cuda_stream;
};

// ----------------------------------------------------------------------------
// HELPER FUNCTIONS
// ----------------------------------------------------------------------------
std::vector<sl::uint2> cvt(const BBox& bbox_in) {
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x1, bbox_in.y1);
    bbox_out[1] = sl::uint2(bbox_in.x2, bbox_in.y1);
    bbox_out[2] = sl::uint2(bbox_in.x2, bbox_in.y2);
    bbox_out[3] = sl::uint2(bbox_in.x1, bbox_in.y2);
    return bbox_out;
}

// [NEW] 마커 설정 로드 함수
void loadMarkerConfig(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[Config] Warning: " << filename << " not found. Using default threshold 0.05m and no markers." << std::endl;
        return;
    }

    int count = 0;
    file >> count; // 1. 마커 갯수
    file >> drift_threshold; // 2. 오차 허용 범위 (m)

    std::cout << "[Config] Drift Threshold set to: " << drift_threshold << " m" << std::endl;

    int id;
    float x, y, z;
    for (int i = 0; i < count; ++i) {
        if (file >> id >> x >> y >> z) {
            sl::Transform tf;
            tf.setIdentity();
            // 마커의 월드 좌표 설정 (회전은 월드 축과 동일하다고 가정)
            tf.setTranslation(sl::float3(x, y, z));
            known_markers_map[id] = tf;
            printf("[Config] Loaded Marker ID %d at World(%.2f, %.2f, %.2f)\n", id, x, y, z);
        }
    }
    file.close();
}

void sendMeshChunkBinaryWithIndex(const sl::Chunk& chunk, uint32_t chunk_index) {
    if (!socket_client.opened()) return;

    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&chunk_index), reinterpret_cast<uint8_t*>(&chunk_index) + 4);

    uint32_t v_size = chunk.vertices.size();
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&v_size), reinterpret_cast<uint8_t*>(&v_size) + 4);
    buffer.insert(buffer.end(), reinterpret_cast<const uint8_t*>(chunk.vertices.data()), reinterpret_cast<const uint8_t*>(chunk.vertices.data()) + sizeof(sl::float3) * v_size);

    uint32_t t_size = chunk.triangles.size();
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&t_size), reinterpret_cast<uint8_t*>(&t_size) + 4);
    buffer.insert(buffer.end(), reinterpret_cast<const uint8_t*>(chunk.triangles.data()), reinterpret_cast<const uint8_t*>(chunk.triangles.data()) + sizeof(sl::uint3) * t_size);

    uint32_t c_size = chunk.colors.size();
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&c_size), reinterpret_cast<uint8_t*>(&c_size) + 4);
    buffer.insert(buffer.end(), reinterpret_cast<const uint8_t*>(chunk.colors.data()), reinterpret_cast<const uint8_t*>(chunk.colors.data()) + sizeof(sl::uchar3) * c_size);

    auto bin_data = std::make_shared<std::string>(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    socket_client.socket()->emit("mesh_chunk", bin_data);
}

// ----------------------------------------------------------------------------
// ArUco Logic (Updated: Ignore Y-axis for Drift Check)
// ----------------------------------------------------------------------------
bool detectAndRelocate(CameraData& it, cv::Mat& cv_rgba, cv::Matx33d& cam_mtx, aruco::Dictionary dict) {
    cv::Mat image_rgb;
    cv::cvtColor(cv_rgba, image_rgb, cv::COLOR_RGBA2RGB);

    vector<int> ids;
    vector<vector<cv::Point2f>> corners;
    aruco::detectMarkers(image_rgb, dict, corners, ids);

    it.tracking_state = it.zed.getPosition(it.pose);

    if (ids.empty()) return false;

    for (size_t i = 0; i < ids.size(); ++i) {
        int id = ids[i];
        if (known_markers_map.find(id) == known_markers_map.end()) continue;
 
        // 1. ArUco 포즈 추정 (OpenCV 표준 출력값)
        vector<cv::Vec3d> rvecs, tvecs;
        cv::Matx<float, 4, 1> dist_coeffs = cv::Vec4f::zeros();
        vector<vector<cv::Point2f>> current_corner = { corners[i] };
        aruco::estimatePoseSingleMarkers(current_corner, marker_size, cam_mtx, dist_coeffs, rvecs, tvecs);

        // 2. OpenCV tvec/rvec을 ZED Transform으로 직결
        sl::Transform marker_in_cam;
        marker_in_cam.setTranslation(sl::float3(tvecs[0](0), tvecs[0](1), -tvecs[0](2)));
        sl::float3 e = it.pose.getEulerAngles();
        marker_in_cam.setRotationVector(sl::float3(e.x, e.y, e.z));
        marker_in_cam.inverse();

        // 3. 월드 기준 카메라 위치 계산
        // T_world_cam = T_world_marker * (T_cam_marker)^-1
        sl::Transform marker_in_world = known_markers_map[id];
        sl::Transform cam_pose_calculated = marker_in_world * marker_in_cam;

        // 4. 오차 확인 (XZ 평면 거리만 계산)
        bool need_reset = false;
        
        if (!is_relocated) {
            cout << "[ArUco] First Identification (ID: " << id << ")" << endl;
            need_reset = true;
        }
        /*
        else {
            if (it.tracking_state == POSITIONAL_TRACKING_STATE::OK) {
                sl::float3 zed_pos = it.pose.getTranslation();
                sl::float3 aruco_pos = cam_pose_calculated.getTranslation();
                
                cout << "Zed x: " + to_string(zed_pos.x) + ", Zed z: " + to_string(zed_pos.z) <<endl;
                cout << "ArUco x: " + to_string(aruco_pos.x) + ", Zed z: " + to_string(aruco_pos.z) <<endl;

                float dx = zed_pos.x - aruco_pos.x;
                float dz = zed_pos.z - aruco_pos.z;
                float dist_error_xz = sqrt(dx * dx + dz * dz);

                if (dist_error_xz > drift_threshold) {
                    printf("[Drift] XZ Error: %.3fm (ID: %d)\n", dist_error_xz, id);
                    need_reset = true;
                }
            }
        }*/

        // 5. 위치 보정 실행 (수정된 부분)
        if (need_reset) {
            it.zed.resetPositionalTracking(cam_pose_calculated);
            if (!is_relocated) is_relocated = true;
            return true;
        }
    }
    return false;
}

// ----------------------------------------------------------------------------
// THREAD LOGIC
// ----------------------------------------------------------------------------
void processSingleCamera(CameraData& it, Yolo& detector, RuntimeParameters& rt_p, CUstream& stream, aruco::Dictionary dict) {
    if (it.zed.grab(rt_p) != ERROR_CODE::SUCCESS) return;

    it.zed.retrieveImage(it.im_left, sl::VIEW::LEFT, sl::MEM::GPU, sl::Resolution(0, 0), stream);
    it.zed.retrieveMeasure(it.point_cloud, MEASURE::XYZRGBA, MEM::GPU, it.point_cloud.getResolution());

    // 1. Master: ArUco Search & Relocalization
    // Master 카메라는 항상 마커를 주시하여 오차를 보정함
    if (it.first_camera) {
        sl::Mat img_cpu;
        it.zed.retrieveImage(img_cpu, VIEW::LEFT, MEM::CPU);
        cv::Mat cv_img = cv::Mat(img_cpu.getHeight(), img_cpu.getWidth(), CV_8UC4, img_cpu.getPtr<sl::uchar1>(sl::MEM::CPU));

        detectAndRelocate(it, cv_img, it.camera_matrix, dict);
    }

    // 2. Global Blocking: 마커 인식 전까지 모든 카메라 동작 중지
    if (!is_relocated) return;

    // 3. Slave: Object Detection
    if (!it.first_camera && object_detect_status) {
        std::vector<sl::CustomBoxObjectData> objects_in;
        auto display_resolution = it.zed.getCameraInformation().camera_configuration.resolution;

        {
            std::lock_guard<std::mutex> lock(detector_mutex);
            auto detections = detector.run(it.im_left, display_resolution.height, display_resolution.width, CONF_THRESH);

            for (auto& itd : detections) {
                sl::CustomBoxObjectData tmp;
                tmp.unique_object_id = sl::generate_unique_id();
                tmp.probability = itd.prob;
                tmp.label = (int)itd.label;
                tmp.bounding_box_2d = cvt(itd.box);
                tmp.is_grounded = ((int)itd.label == 0);
                objects_in.push_back(tmp);
            }
        }
        it.zed.ingestCustomBoxObjects(objects_in);
        sl::CustomObjectDetectionRuntimeParameters customObjectTracker_rt;
        it.zed.retrieveCustomObjects(it.objects, customObjectTracker_rt);

        auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - it.obj_last).count();
        if (duration > 100 && !it.objects.object_list.empty()) {
            std::string json;
            json.reserve(2048);
            json += "[";
            bool first = true;
            for (auto& object : it.objects.object_list) {
                if (!first) json += ",";
                first = false;
                char buf[128];
                vector<sl::float3> bb_info = object.bounding_box;
                float cx = object.position.x;
                float cy = object.position.y;
                for (size_t i = 0; i < bb_info.size(); i++) {
                    float rel_x = bb_info[i].x - cx;
                    float rel_y = bb_info[i].y - cy;
                    // Note: Rotation logic (hardcoded 90 deg?) kept as per original request
                    float rot_x = -rel_y;
                    float rot_y = -rel_x;
                    float final_x = rot_x + cx;
                    float final_y = rot_y + cy;
                    float final_z = bb_info[i].z;
                    std::snprintf(buf, sizeof(buf), "[%d,%.3f,%.3f,%.3f]", object.id, final_x, final_y, final_z);
                    json += buf;
                    if (i < bb_info.size() - 1) json += ",";
                }
            }
            json += "]";
            if (json.length() > 2) {
                socket_client.socket()->emit("obj_info", std::make_shared<std::string>(json));
                it.obj_last = chrono::high_resolution_clock::now();
            }
        }
    }
    else if (!object_detect_status) {
        it.objects.object_list.clear();
    }

    // 5. Send Pose (Master Only)
    if (it.tracking_state == POSITIONAL_TRACKING_STATE::OK && it.first_camera) {
        auto now_tp = std::chrono::steady_clock::now();
        auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_tp - it.kLastPoseSend).count();
        if (dt_ms >= 250) {
            sl::float3 t = it.pose.getTranslation();
            sl::float3 e = it.pose.getEulerAngles();
            char buf[128];
            std::snprintf(buf, sizeof(buf), "{\"x\":%.5f,\"y\":%.5f,\"z\":%.5f,\"ax\":%.5f,\"ay\":%.5f,\"az\":%.5f}", t.x, t.y, t.z, e.x, e.y, e.z);
            socket_client.socket()->emit("camera_pose", std::make_shared<std::string>(buf));
            it.kLastPoseSend = now_tp;
        }
    }

    // 6. Recording
    if (record_status && !it.is_record) {
        auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm tm; localtime_r(&t, &tm);
        char buf[32];
        std::sprintf(buf, "%02d%02d%02d%02d%02d%02d", tm.tm_year % 100, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
        std::string svo_filename = "save_data/" + std::string(sl::toString(it.zed.getCameraInformation().camera_model)) +
            "_SN" + std::to_string(it.zed.getCameraInformation().serial_number) + "_" + buf + ".svo2";
        sl::RecordingParameters rec_params;
        rec_params.video_filename.set(svo_filename.c_str());
        rec_params.compression_mode = sl::SVO_COMPRESSION_MODE::H265;
        it.zed.enableRecording(rec_params);
        it.is_record = true;
        cout << "[Rec Start] " << it.info << endl;
    }
    else if (!record_status && it.is_record) {
        it.zed.disableRecording();
        it.is_record = false;
        cout << "[Rec Stop] " << it.info << endl;
    }
}

// ----------------------------------------------------------------------------
// MAIN & LOOP
// ----------------------------------------------------------------------------
void run(vector<CameraData>& zeds) {
    RuntimeParameters rt_p;
    rt_p.measure3D_reference_frame = REFERENCE_FRAME::WORLD;

    SpatialMappingParameters mapping_params;
    mapping_params.map_type = SpatialMappingParameters::SPATIAL_MAP_TYPE::MESH;
    mapping_params.set(sl::SpatialMappingParameters::MAPPING_RESOLUTION::LOW);
    mapping_params.set(sl::SpatialMappingParameters::MAPPING_RANGE::SHORT);
    mapping_params.use_chunk_only = false;

    std::ifstream md_file("models.txt");
    if (!md_file.is_open()) return;
    std::string models_name;
    std::getline(md_file, models_name);

    std::string engine_name = models_name;
    Yolo detector;
    detector.init(engine_name);

    auto dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_1000);

    while (!quit) {
        if (reset_status) {
            is_relocated = false; // 리셋 시 다시 마커 찾기 모드로
            for (auto& it : zeds) {
                it.zed.disablePositionalTracking();
                PositionalTrackingParameters tp;
                tp.enable_area_memory = true;
                tp.mode = sl::POSITIONAL_TRACKING_MODE::GEN_3;
                tp.initial_world_transform = it.tf;
                it.zed.enablePositionalTracking(tp);

                if (it.first_camera) {
                    it.zed.disableSpatialMapping();
                    it.request_new_mesh = true;
                    it.wait_for_mapping = true;
                    it.map.clear();
                }
            }
            reset_status = false;
        }

        if (pose_update) {
            for (auto& it : zeds) {
                if (it.first_camera) it.tf = tf[0];
                else it.tf = tf[1];
                it.zed.resetPositionalTracking(it.tf);
            }
            pose_update = false;
        }

        std::vector<std::future<void>> futures;
        for (auto& it : zeds) {
            futures.push_back(std::async(std::launch::async, processSingleCamera,
                std::ref(it), std::ref(detector), std::ref(rt_p), std::ref(it.zed_cuda_stream), std::ref(dictionary)));
        }
        for (auto& f : futures) f.wait();

        for (auto& it : zeds) {
            it.im_left.updateCPUfromGPU(it.zed_cuda_stream);
            cv::Mat draw_img = slMat2cvMat(it.im_left);
            std::string title = "ZED" + std::to_string(it.id);
            cv::imshow(title, draw_img);
        }

        // Mapping은 Master가 Relocate 된 후에만 수행
        if (is_relocated) {
            for (auto& it : zeds) {
                if (it.first_camera && it.tracking_state == POSITIONAL_TRACKING_STATE::OK) {
                    if (it.wait_for_mapping) {
                        // 중요: Mapping 시작 시점의 포즈 유지 (resetPositionalTracking은 detectAndRelocate에서 처리됨)
                        it.zed.enableSpatialMapping(mapping_params);
                        it.wait_for_mapping = false;
                    }
                    else {
                        if (it.request_new_mesh) {
                            auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - it.ts_last).count();
                            if (duration > 200) {
                                it.zed.requestSpatialMapAsync();
                                it.request_new_mesh = false;
                            }
                        }
                        if (it.zed.getSpatialMapRequestStatusAsync() == ERROR_CODE::SUCCESS && !it.request_new_mesh) {
                            it.zed.retrieveSpatialMapAsync(it.map);
                            for (size_t i = 0; i < it.map.chunks.size(); ++i) {
                                if (it.map.chunks[i].has_been_updated) {
                                    sendMeshChunkBinaryWithIndex(it.map.chunks[i], static_cast<uint32_t>(i) + it.mapping_index);
                                }
                            }
                            it.request_new_mesh = true;
                            it.ts_last = chrono::high_resolution_clock::now();
                        }
                    }
                }
            }
        }
        sl::sleep_ms(1);
    }
}

// ----------------------------------------------------------------------------
// MAIN FUNCTION
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    // 1. Socket Connection
    std::ifstream ip_file("ipadress.txt");
    std::string ipadress = "http://127.0.0.1:3000";
    if (ip_file.is_open()) std::getline(ip_file, ipadress);

    socket_client.connect(ipadress);
    std::cout << "[Socket] Connecting to " << ipadress << "..." << std::endl;

    // [NEW] Load Markers Config
    loadMarkerConfig("markers.txt");

    // 2. Camera Discovery
    auto zed_infos = Camera::getDeviceList();
    int nb_zeds = zed_infos.size();
    if (nb_zeds == 0) {
        cout << "Error: No ZED detected\n";
        return 1;
    }

    vector<CameraData> zeds(nb_zeds);

    // 3. Initialize Cameras
    InitParameters init_params;
    init_params.depth_mode = DEPTH_MODE::NEURAL;
    init_params.coordinate_units = UNIT::METER;
    init_params.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_params.camera_fps = 30;

    int nb_zed_open = 0;
    for (int i = 0; i < nb_zeds; i++) {
        init_params.input.setFromCameraID(zed_infos[i].id);
        ERROR_CODE err = zeds[i].zed.open(init_params);

        if (err == ERROR_CODE::SUCCESS) {
            nb_zed_open++;
            zeds[i].id = i;

            // Memory Allocation
            zeds[i].point_cloud.alloc(Resolution(512, 288), MAT_TYPE::F32_C4, MEM::GPU);
            auto camCalib = zeds[i].zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam;
            zeds[i].im_left.alloc(camCalib.image_size, MAT_TYPE::U8_C4);

            zeds[i].camera_matrix(0, 0) = camCalib.fx;
            zeds[i].camera_matrix(1, 1) = camCalib.fy;
            zeds[i].camera_matrix(0, 2) = camCalib.cx;
            zeds[i].camera_matrix(1, 2) = camCalib.cy;

            // --- MODEL & ROLE SETUP ---
            sl::MODEL cam_model = zeds[i].zed.getCameraInformation().camera_model;

            if (cam_model == sl::MODEL::ZED_X) {
                zeds[i].first_camera = true;    // MASTER
                zeds[i].tf = tf[0];             // Origin TF
                cout << "[Setup] ZED X detected (MASTER) - ID: " << i << endl;
            }
            else {
                zeds[i].first_camera = false;   // SLAVE
                zeds[i].tf = tf[1];             // Offset TF
                cout << "[Setup] ZED 2/2i detected (SLAVE) - ID: " << i << endl;
            }

            // Tracking Setup
            PositionalTrackingParameters tracking_params;
            tracking_params.enable_area_memory = true;
            tracking_params.mode = sl::POSITIONAL_TRACKING_MODE::GEN_3;
            tracking_params.initial_world_transform = zeds[i].tf; // Important!
            zeds[i].zed.enablePositionalTracking(tracking_params);

            // Object Detection Setup
            if (!zeds[i].first_camera) {
                ObjectDetectionParameters obj_params;
                obj_params.enable_tracking = true;
                obj_params.detection_model = OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
                zeds[i].zed.enableObjectDetection(obj_params);
            }

            zeds[i].info = "ZED_" + std::to_string(i);
            zeds[i].mapping_index = i * 10000;
            zeds[i].zed_cuda_stream = zeds[i].zed.getCUDAStream();
        }
    }

    if (nb_zed_open != nb_zeds) {
        cout << "Error: Could not open all cameras." << endl;
        return 1;
    }

    // 4. Socket Events
    socket_client.socket()->on("record_status", sio::socket::event_listener_aux([&](string const& name, sio::message::ptr const& data, bool isAck, sio::message::list& ack_resp) {
        record_status = !record_status;
        cout << "[Event] Record Status: " << record_status << endl;
        }));

    socket_client.socket()->on("object_detect", sio::socket::event_listener_aux([&](string const& name, sio::message::ptr const& data, bool isAck, sio::message::list& ack_resp) {
        object_detect_status = !object_detect_status;
        cout << "[Event] Object Detect: " << (object_detect_status ? "ON" : "OFF") << endl;
        }));

    socket_client.socket()->on("reset_mesh_chunk", sio::socket::event_listener_aux([&](string const& name, sio::message::ptr const& data, bool isAck, sio::message::list& ack_resp) {
        reset_status = true;
        cout << "[Event] Reset Triggered" << endl;
        }));

    socket_client.socket()->on("set_camera", sio::socket::event_listener_aux([&](string const& name, sio::message::ptr const& data, bool isAck, sio::message::list& ack_resp) {
        if (data->get_flag() != sio::message::flag_string) return;
        std::string s = data->get_string();
        float x1, y1, x2, y2;
        if (sscanf(s.c_str(), "{%f,%f,%f,%f}", &x1, &y1, &x2, &y2) == 4) {
            tf[0] = sl::Transform(sl::Rotation(), sl::Translation(x1, 0, y1));
            tf[1] = sl::Transform(sl::Rotation(), sl::Translation(x2, 0, y2));
            pose_update = true;
            cout << "[Event] Camera TF Updated" << endl;
        }
        }));

    // 5. Run Loop
    std::cout << "Starting Threads... Press Ctrl+C to exit." << std::endl;
    std::thread runner(run, std::ref(zeds));
    runner.join();

    // 6. Cleanup
    for (auto& it : zeds) {
        it.point_cloud.free();
        it.im_left.free();
        it.zed.close();
    }
    return 0;
}
