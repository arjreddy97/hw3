import cv2
import numpy as np

image_path = "cube_screenshot.png"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"❌ Image not found: {image_path}")

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, str(len(points)), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Click visible cube corners", img)

cv2.imshow("Click visible cube corners", img)
cv2.setMouseCallback("Click visible cube corners", mouse_callback)

print("🖱 Click 4 or more **visible** corners of the cube in the image.")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) < 4:
    print(f"⚠️ Only {len(points)} points clicked. Need at least 4 for pose estimation.")
else:
    np.savez("cube_2d_points.npz", image_points=np.array(points, dtype=np.float32))
    print("✅ 2D cube points saved to cube_2d_points.npz")
