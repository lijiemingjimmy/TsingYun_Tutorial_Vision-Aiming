import cv2
import numpy as np
img = cv2.imread('debug_crop.png')
# Note cv2.imread loads in BGR
min_c = np.min(img, axis=2)
print("min_c max:", min_c.max(), "min_c min:", min_c.min(), "min_c mean:", min_c.mean())
