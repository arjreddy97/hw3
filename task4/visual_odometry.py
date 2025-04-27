import cv2
import numpy as np

class VisualOdometry:
    def __init__(self, K):
        self.K = K
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_des = None
        self.cur_pose = np.eye(4)

    def process_frame(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        if self.prev_kp is None:
            self.prev_kp, self.prev_des = kp, des
            return self.cur_pose

        matches = self.bf.match(des, self.prev_des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 8:
            return self.cur_pose  # Not enough matches

        pts1 = np.float32([kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.prev_kp[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return self.cur_pose

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        self.cur_pose = self.cur_pose @ np.linalg.inv(T)

        self.prev_kp, self.prev_des = kp, des
        return self.cur_pose
