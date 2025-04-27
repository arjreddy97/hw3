import cv2
import numpy as np
import glob

CHECKERBOARD = (6, 9)
SQUARE_SIZE = 0.025
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints, imgpoints = [], []
images = glob.glob('frame_*.png')
gray = None
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(200)
cv2.destroyAllWindows()

if not objpoints or not imgpoints or gray is None:
    print("❌ No valid checkerboard images found.")
    exit(1)

ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("✅ Calibration Successful!")
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)
np.savez('camera_calib_data.npz', camera_matrix=mtx, dist_coeff=dist)
