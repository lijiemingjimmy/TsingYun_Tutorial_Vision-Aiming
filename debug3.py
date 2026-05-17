import sys
from pathlib import Path
sys.path.append('tasks/task2-detector/src')
import numpy as np
import cv2
from detector import detect_mnist_board, detect_bbox, crop_bbox
from model import preprocess_mnist_crop

from tests.python.test_task2_detector import _take_game_screenshots

screenshots = _take_game_screenshots()
for i, img in enumerate(screenshots):
    print(f"--- Image {i} ---")
    candidates = detect_bbox(img, 200)
    crops = crop_bbox(img, candidates)
    for crop in crops:
        tensor = preprocess_mnist_crop(crop)
        print("Tensor:")
        for row in range(28):
            print(" ".join(f"{x:5.2f}" for x in tensor[0, 0, row, :]))
        break
    break
