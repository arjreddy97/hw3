import numpy as np
import matplotlib.pyplot as plt
from dataset import KITTIDataset
from visual_odometry import VisualOdometry

# KITTI camera intrinsic matrix (sequence 00)
K = np.array([[718.856, 0, 607.1928],
              [0, 718.856, 185.2157],
              [0, 0, 1]])

# ✅ Use your actual dataset path here
dataset = KITTIDataset("/home/areddy/kitti/dataset/sequences/00/image_0")
vo = VisualOdometry(K)

trajectory = []

for i in range(len(dataset)):
    img = dataset[i]
    pose = vo.process_frame(img)
    x, z = pose[0, 3], pose[2, 3]
    trajectory.append((x, z))

trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.title("Estimated Trajectory")
plt.xlabel("x [m]")
plt.ylabel("z [m]")
plt.grid()
plt.axis("equal")
plt.savefig("output/trajectory.png")
plt.show()
