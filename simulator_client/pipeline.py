"""Safe integrated pipeline used by the simulator runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator_client.kalman_tracker_bindings import KalmanTracker
from simulator_client.protocol import AimMessage, Matrix3x3
from detector import BoundingBox, Detection, detect_mnist_board


@dataclass(frozen=True)
class PipelineResult:
    aim: AimMessage
    used_fallback: bool
    reason: str


class FallbackPipeline:
    def __init__(
        self,
        latency: float = 1.0,
        target_depth: float = 10.0,
        threshold: int = 200,
        board_width_meters: float = 0.4,
        board_height_meters: float = 0.2,
        latency_multiplier: float = 1.0,
    ) -> None:
        self.latency = latency
        self.target_depth = target_depth
        self.threshold = threshold
        self.board_width_meters = board_width_meters
        self.board_height_meters = board_height_meters
        self.latency_multiplier = latency_multiplier
        # TODO: fine-tune your arguments here
        self.tracker = KalmanTracker(process_noise=0.5, measurement_noise=0.05)
        self._last_time: float | None = None
        self._lost_count: int = 0
        self._locked_class_id: int | None = None

    def process_rgb_image(self, image: np.ndarray, camera_matrix: Matrix3x3, timestamp: float = 0.0) -> PipelineResult:
        try:
            dt = timestamp - self._last_time if self._last_time is not None else 0.05
            self._last_time = timestamp
            dt = max(0.001, min(dt, 1.0))

            detections = detect_mnist_board(image, threshold=self.threshold)

            # Filter to confident detections
            good = [d for d in detections if d.class_id >= 0 and d.confidence >= 0.2]
            if not good:
                good = detections  # fallback: use any detection

            if not good:
                self._lost_count += 1
                if self.tracker.is_tracking and self._lost_count <= 20:
                    px, py, pz = self.tracker.predict(self.latency * self.latency_multiplier)
                    pz = max(1.0, min(pz, 200.0))
                    return PipelineResult(
                        AimMessage(float(px), float(py), float(pz)),
                        used_fallback=False, reason="coasting",
                    )
                self.tracker.reset()
                self._locked_class_id = None
                return self._center_fallback(image, camera_matrix, f"no target in {len(detections)} detections")

            self._lost_count = 0

            # Estimate 3D positions for all detections
            positions = [self._estimate_position(d, camera_matrix) for d in good]
            
            h, w = image.shape[:2]
            center_x, center_y = w / 2.0, h / 2.0
            max_dist_sq = (w/2)**2 + (h/2)**2

            best_idx = -1
            best_score = -9999.0
            
            # 获取当前追踪器的预测位置，用于计算粘性
            pred_pos = self.tracker.get_position() if self.tracker.is_tracking else None

            for i, d in enumerate(good):
                dist_sq = (d.bbox.center[0] - center_x)**2 + (d.bbox.center[1] - center_y)**2
                dist_norm = dist_sq / max_dist_sq  # 归一化距离 [0, 1]
                
                # 统一评分公式：数字大加分，距离远扣分，置信度高加分
                # 距离最远扣 5 分，意味着屏幕边缘的 8 等同于中心的 3
                score = d.class_id - (dist_norm * 5.0) + (d.confidence * 2.0)
                
                # 追踪粘性 (Hysteresis) 机制
                if self.tracker.is_tracking and self._locked_class_id == d.class_id and pred_pos is not None:
                    cur_x, cur_y, cur_z = positions[i]
                    pred_x, pred_y, pred_z = pred_pos
                    # 检查 3D 物理距离，误差在 2.0 米内认为是同一个追踪目标
                    if (cur_x - pred_x)**2 + (cur_y - pred_y)**2 + (cur_z - pred_z)**2 < 4.0:
                        score += 4.0  # 给予 4.0 的粘性分！新靶子必须比当前靶子“香” 4 分以上才能抢走锁定
                        
                if score > best_score:
                    best_score = score
                    best_idx = i

            target_d = good[best_idx]

            # 目标发生了实质性切换，或者第一次追踪，需要重置滤波器
            if not self.tracker.is_tracking or self._locked_class_id != target_d.class_id:
                self.tracker.reset()
                self._locked_class_id = target_d.class_id

            cur_x, cur_y, cur_z = positions[best_idx]
            self.tracker.update(cur_x, cur_y, cur_z, dt)
            pred_x, pred_y, pred_z = self.tracker.predict(self.latency * self.latency_multiplier)


            # Clamp
            pred_z = max(1.0, min(pred_z, 200.0))
            max_xy = abs(pred_z) * 2.0
            pred_x = max(-max_xy, min(pred_x, max_xy))
            pred_y = max(-max_xy, min(pred_y, max_xy))

            # Build detection visualization data
            dets_payload = []
            for d, (px, py, pz) in zip(good, positions):
                dets_payload.append({
                    "classId": d.class_id,
                    "confidence": d.confidence,
                    "bbox": {
                        "x": d.bbox.x, "y": d.bbox.y,
                        "width": d.bbox.width, "height": d.bbox.height,
                    },
                    "corners": [
                        [float(d.corners[0][0]), float(d.corners[0][1])],
                        [float(d.corners[1][0]), float(d.corners[1][1])],
                        [float(d.corners[2][0]), float(d.corners[2][1])],
                        [float(d.corners[3][0]), float(d.corners[3][1])],
                    ],
                    "position": {"x": float(px), "y": float(py), "z": float(pz)},
                })

            return PipelineResult(
                AimMessage(
                    float(pred_x), float(pred_y), float(pred_z),
                    detections=dets_payload,
                ),
                used_fallback=False, reason="tracking",
            )
        except NotImplementedError as exc:
            return self._center_fallback(image, camera_matrix, f"student function not implemented: {exc}")
        except Exception as exc:
            return self._center_fallback(image, camera_matrix, f"pipeline error: {exc}")

    def _estimate_position(self, detection: Detection, camera_matrix: Matrix3x3) -> tuple[float, float, float]:
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]

        bbox_w = max(detection.bbox.width, 1.0)
        z_est = fx * self.board_width_meters / bbox_w
        z_est = max(1.0, min(z_est, 200.0))

        u, v = detection.bbox.center
        ray_x = (u - cx) / fx
        ray_y = (v - cy) / fy
        return (ray_x * z_est, ray_y * z_est, z_est)

    def _center_fallback(self, image: np.ndarray, camera_matrix: Matrix3x3, reason: str) -> PipelineResult:
        height, width = image.shape[:2]
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]
        u, v = (width - 1) * 0.5, (height - 1) * 0.5
        ray_x = (u - cx) / fx
        ray_y = (v - cy) / fy
        px, py, pz = ray_x * self.target_depth, ray_y * self.target_depth, self.target_depth
        return PipelineResult(AimMessage(float(px), float(py), float(pz)), used_fallback=True, reason=reason)
