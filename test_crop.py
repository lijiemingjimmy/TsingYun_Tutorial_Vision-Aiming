import cv2
img = cv2.imread('debug_crop.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("max:", gray.max(), "min:", gray.min(), "mean:", gray.mean())
