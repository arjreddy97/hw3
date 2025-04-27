simulate_checkerboard_frames.py
import cv2
import numpy as np

img = cv2.imread("checkerboard.png")
if img is None:
    raise FileNotFoundError("checkerboard.png not found")

h, w = img.shape[:2]
pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

for i in range(10):
    dx = np.random.randint(-40, 40)
    dy = np.random.randint(-30, 30)
    pts2 = np.float32([
        [0 + dx, 0 + dy],
        [w + dx//2, 0 - dy],
        [0 - dx, h + dy//2],
        [w + dx//3, h + dy//3]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (w, h))
    filename = f"frame_{i}.png"
    cv2.imwrite(filename, warped)
    print(f"✅ Saved {filename}")
