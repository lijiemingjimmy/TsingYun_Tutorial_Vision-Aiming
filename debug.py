import sys
from pathlib import Path
sys.path.append('tasks/task2-detector/src')
import numpy as np
import cv2
from detector import detect_mnist_board, detect_bbox, crop_bbox
from model import classify_mnist_digit

from tests.python.test_task2_detector import _take_game_screenshots

screenshots = _take_game_screenshots()
for i, img in enumerate(screenshots):
    print(f"--- Image {i} ---")
    candidates = detect_bbox(img, 200)
    print(f"Bbox candidates: {candidates}")
    crops = crop_bbox(img, candidates)
    for crop in crops:
        digit, conf = classify_mnist_digit(crop)
        print(f"Crop classification: digit={digit}, conf={conf}")
