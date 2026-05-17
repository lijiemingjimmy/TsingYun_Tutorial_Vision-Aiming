"""Task 2 MNIST-board detector helpers with student TODO extension points.

This file belongs to Task 2. The simulator runner imports it so that a Task 2
implementation can be tested both offline and inside the Unity simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from simulator_client.protocol import Matrix3x3
from model import classify_mnist_digit

Point2D = tuple[float, float]
CornerSet = tuple[Point2D, Point2D, Point2D, Point2D]
RgbPixel = tuple[int, int, int]
ImageLike = np.ndarray
WARP_OUTPUT_SIZE = 128
MNIST_INNER_RATIO = 0.69


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point2D:
        return (self.x + self.width * 0.5, self.y + self.height * 0.5)


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: BoundingBox
    corners: CornerSet
    rvec: object | None = None
    tvec: object | None = None


def _bbox_from_corners(corners: Sequence[Point2D]) -> BoundingBox:
    if len(corners) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")

    xs = [float(point[0]) for point in corners]
    ys = [float(point[1]) for point in corners]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return BoundingBox(
        x=min_x,
        y=min_y,
        width=max_x - min_x + 1.0,
        height=max_y - min_y + 1.0,
    )


def _crop_bounds(corners: Sequence[Point2D], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    bbox = _bbox_from_corners(corners)
    x0 = max(0, min(image_width, int(np.floor(bbox.x))))
    y0 = max(0, min(image_height, int(np.floor(bbox.y))))
    x1 = max(0, min(image_width, int(np.ceil(bbox.x + bbox.width))))
    y1 = max(0, min(image_height, int(np.ceil(bbox.y + bbox.height))))
    return x0, y0, x1, y1


def crop_bbox(image: np.ndarray, corner_candidates: Sequence[Sequence[Point2D]]) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    for corners in corner_candidates:
        if len(corners) != 4:
            continue

        # `corners` are expected in LU, RU, RD, LD order.
        src = np.array(corners, dtype=np.float32)

        # Keep 15% border on each side, so the central 70% region contains MNIST.
        margin = (1.0 - MNIST_INNER_RATIO) * 0.5 * (WARP_OUTPUT_SIZE - 1)
        dst = np.array(
            [
                [margin, margin],
                [WARP_OUTPUT_SIZE - 1 - margin, margin],
                [WARP_OUTPUT_SIZE - 1 - margin, WARP_OUTPUT_SIZE - 1 - margin],
                [margin, WARP_OUTPUT_SIZE - 1 - margin],
            ],
            dtype=np.float32,
        )

        perspective = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            image,
            perspective,
            (WARP_OUTPUT_SIZE, WARP_OUTPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        crops.append(warped)
    return crops


def order_corners(corners: Sequence[Point2D]) -> CornerSet:
    sorted_x=sorted(corners,key=lambda p:p[0])
    left_points=sorted_x[:2]
    right_points=sorted_x[2:]
    left_points=sorted(left_points,key=lambda p:p[1])
    top_left=left_points[0]
    bottom_left=left_points[1]
    right_points=sorted(right_points,key=lambda p:p[1])
    top_right=right_points[0]
    bottom_right=right_points[1]
    return top_left,top_right,bottom_right,bottom_left



def detect_bbox(image: ImageLike, threshold: int = 200) -> list[CornerSet]:
    image_array=np.array(image,dtype=np.uint8)
    r=image_array[:,:,0]
    g=image_array[:,:,1]
    b=image_array[:,:,2]
    red_mask = ((r > threshold) & (r > g + 20) & (r > b + 20)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corner_candidates = []
    for contour in contours:
        area=cv2.contourArea(contour)
        if area<100:
            continue
        epsilon=0.04*cv2.arcLength(contour,True)
        polygon=cv2.approxPolyDP(contour,epsilon,closed=True)
        if len(polygon)!=4:
            continue
        pts = [(float(p[0][0]), float(p[0][1])) for p in polygon]
        corners = order_corners(pts)
        corner_candidates.append(corners)
        
    return corner_candidates


def detect_mnist_board(image: ImageLike, threshold: int = 200) -> list[Detection]:

    corner_candidates = detect_bbox(image, threshold)
    crops = crop_bbox(image, corner_candidates)
    
    detections = []
    for crop, corners in zip(crops, corner_candidates):
        digit, confidence = classify_mnist_digit(crop)
        
        if confidence > 0.4:
            bbox = _bbox_from_corners(corners)
            
            detections.append(Detection(
                class_id=digit,
                confidence=float(confidence),
                bbox=bbox,
                corners=corners
            ))
            
    return detections


def solve_pnp(
    detections: Sequence[Detection],
    camera_matrix: Matrix3x3,
    board_width_meters: float,
    board_height_meters: float,
    dist_coeffs: Sequence[float] | None = None,
) -> list[Detection]:

    half_width = board_width_meters / 2.0
    half_height = board_height_meters / 2.0
    object_points = np.array([
        [-half_width, -half_height, 0],
        [ half_width, -half_height, 0],
        [ half_width,  half_height, 0],
        [-half_width,  half_height, 0],
    ], dtype=np.float32)

    camera_array = np.array(camera_matrix, dtype=np.float64)
    
    if dist_coeffs is None:
        dist_array = np.zeros(5, dtype=np.float64)
    else:
        dist_array = np.array(dist_coeffs, dtype=np.float64)

    result = []
    for detection in detections:
        image_points = np.array(detection.corners, dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_array, dist_array
        )
        
        if success:
            detection.rvec = rvec
            detection.tvec = tvec
            result.append(detection)
            
    return result

    


