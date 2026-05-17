import json
from pathlib import Path

import cv2
import numpy as np

TASK_ROOT = Path(__file__).resolve().parents[1]

# TODO(student): fill in your camera parameter file from calibrate.py.
# run calibrate.py first and point this path at the generated camera_params.json
# the JSON must contain camera_matrix and dist_coeffs
CAMERA_PARAMS_PATH = TASK_ROOT / "output" / "camera_params.json"

# TODO(student): fill in your own ArUco video path.
# use a video where the marker is clear and not too motion-blurred
# keep the marker dictionary and physical length below consistent with this video
ARUCO_VIDEO_PATH = TASK_ROOT / "data" / "aruco" / "aruco_video.mp4"

# TODO(student): fill in your own ArUco settings.
# choose the same dictionary that was used to print the marker
# measure the black marker side length in meters and store it in MARKER_LENGTH_METERS
ARUCO_DICTIONARY = "DICT_4X4_50"
MARKER_LENGTH_METERS = 0.1

ARUCO_OUTPUT_VIDEO_PATH = TASK_ROOT / "output" / "aruco_result.mp4"

# TODO(student): use this path if you want to render one of the provided OBJ models.
# start with cube.obj while debugging because its shape makes pose errors obvious
# after the pose is stable, switch to another OBJ model from res/models
MODEL_PATH = TASK_ROOT / "res" / "models" / "cube.obj"
OUTPUT_DIR = TASK_ROOT / "output"


def load_camera_params(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32)
    return camera_matrix, dist_coeffs


def load_obj(model_path):
    """Load the vertices and faces from a small OBJ model."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    for raw_line in Path(model_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("v "):
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            continue

        if line.startswith("f "):
            parts = line.split()[1:]
            if len(parts) < 3:
                continue

            indices: list[int] = []
            for token in parts:
                vertex_text = token.split("/")[0]
                if not vertex_text:
                    continue
                obj_index = int(vertex_text)
                if obj_index > 0:
                    indices.append(obj_index - 1)
                else:
                    indices.append(len(vertices) + obj_index)

            if len(indices) < 3:
                continue

            for i in range(1, len(indices) - 1):
                faces.append((indices[0], indices[i], indices[i + 1]))

    return vertices, faces


def get_aruco_dictionary(name):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_markers(frame, dictionary):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)
    return corners, ids


def create_marker_object_points(marker_length_meters):
    half = marker_length_meters * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def _is_valid_pose_result(result):
    if result is None:
        return False
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    rvec, tvec = result
    rvec = np.asarray(rvec)
    tvec = np.asarray(tvec)
    return rvec.size == 3 and tvec.size == 3 and np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))


def estimate_marker_pose(marker_corners, marker_length_meters, camera_matrix, dist_coeffs):
    object_points = create_marker_object_points(marker_length_meters)

    image_points = marker_corners.reshape(4, 2)
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    
    if not success:
        return None, None
    return rvec, tvec


def render_virtual_object(frame, rvec, tvec, camera_matrix, dist_coeffs, vertices, faces):
    if not vertices or not faces:
        return frame
        
    # 1. 转换为 Numpy Array 并计算包围盒
    verts = np.array(vertices, dtype=np.float32)
    min_b = np.min(verts, axis=0)
    max_b = np.max(verts, axis=0)
    center = (max_b + min_b) / 2.0
    size = max_b - min_b
    
    # 2. 将模型中心平移至原点
    verts -= center
    
    # 2.5 旋转模型：让模型“站”起来
    # 大部分 3D 模型（如 african_head）是以 Y 轴为向上的头顶方向
    # 而 ArUco 坐标系是以 Z 轴为垂直于平面的向上方向
    # 所以我们需要绕 X 轴旋转 90 度，让原先的 Y 轴变成 Z 轴
    y = verts[:, 1].copy()
    z = verts[:, 2].copy()
    verts[:, 1] = -z
    verts[:, 2] = y
    
    # 3. 缩放模型：使得模型的最大跨度与 Marker 的真实物理大小匹配
    scale = MARKER_LENGTH_METERS / max(size)
    verts *= scale
    
    # 4. 调整高度：将模型的底部正好放到 Marker 所在的平面 (Z=0)
    # 在标准 ArUco 坐标系中，Z 轴通常垂直于表面向外
    min_z = np.min(verts[:, 2])
    verts[:, 2] -= min_z
    
    # 5. 投影：将三维点投影回当前帧的二维像素坐标
    imgpts, _ = cv2.projectPoints(verts, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # 6. 渲染：将三角面片绘制为绿色线框
    for face in faces:
        pts = np.array([imgpts[face[0]], imgpts[face[1]], imgpts[face[2]]])
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
        
    return frame


def process_frame(frame, dictionary, camera_matrix, dist_coeffs, vertices, faces):
    output = frame.copy()
    corners, ids = detect_markers(frame, dictionary)

    if ids is None or len(ids) == 0:
        return output

    cv2.aruco.drawDetectedMarkers(output, corners, ids)

    for marker_corners, _ in zip(corners, ids):
        rvec, tvec = estimate_marker_pose(
            marker_corners,
            MARKER_LENGTH_METERS,
            camera_matrix,
            dist_coeffs,
        )
        render_virtual_object(
            output,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
            vertices,
            faces,
        )

    return output


def run_aruco_render(dictionary, camera_matrix, dist_coeffs, capture, vertices, faces):
    ok, frame = capture.read()
    if not ok:
        raise SystemExit("Cannot read the first frame from the source.")

    height, width = frame.shape[:2]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(ARUCO_OUTPUT_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (width, height),
    )

    try:
        while ok:
            result = process_frame(frame, dictionary, camera_matrix, dist_coeffs, vertices, faces)
            writer.write(result)
            cv2.imshow("aruco", result)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            ok, frame = capture.read()
    finally:
        writer.release()
        capture.release()
        cv2.destroyAllWindows()

    print(f"Saved video result to: {ARUCO_OUTPUT_VIDEO_PATH}")


def main():
    if not CAMERA_PARAMS_PATH.exists():
        raise SystemExit(f"Camera parameters not found: {CAMERA_PARAMS_PATH}")

    camera_matrix, dist_coeffs = load_camera_params(CAMERA_PARAMS_PATH)
    vertices, faces = load_obj(MODEL_PATH)
    dictionary = get_aruco_dictionary(ARUCO_DICTIONARY)
    capture = cv2.VideoCapture(str(ARUCO_VIDEO_PATH))

    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {ARUCO_VIDEO_PATH}")

    run_aruco_render(dictionary, camera_matrix, dist_coeffs, capture, vertices, faces)


if __name__ == "__main__":
    main()
