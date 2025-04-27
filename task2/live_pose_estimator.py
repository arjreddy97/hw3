import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class LivePoseEstimator(Node):
    def __init__(self):
        super().__init__('live_pose_estimator')
        self.bridge = CvBridge()

        # Load calibration
        calib = np.load('camera_calib_data.npz')
        self.K = calib['camera_matrix']
        self.D = calib['dist_coeff']

        # Load 2D image points (manually labeled on a static image)
        self.cube_2d = np.load('cube_2d_points.npz')['image_points'].astype(np.float32).reshape(-1, 1, 2)

        # Define corresponding 3D cube points in the world
        self.cube_3d = np.array([
            [-0.68 - 0.5, 0.95 - 0.5, 0.0],  # bottom-front-left
            [-0.68 + 0.5, 0.95 - 0.5, 0.0],  # bottom-front-right
            [-0.68 + 0.5, 0.95 + 0.5, 0.0],  # bottom-back-right
            [-0.68 - 0.5, 0.95 + 0.5, 0.0],  # bottom-back-left
            [-0.68 - 0.5, 0.95 - 0.5, 1.0],  # top-front-left
            [-0.68 + 0.5, 0.95 - 0.5, 1.0],  # top-front-right
            [-0.68 + 0.5, 0.95 + 0.5, 1.0],  # top-back-right
            [-0.68 - 0.5, 0.95 + 0.5, 1.0]   # top-back-left
        ], dtype=np.float32).reshape(-1, 1, 3)

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        success, rvec, tvec = cv2.solvePnP(self.cube_3d, self.cube_2d, self.K, self.D)

        if success:
            self.get_logger().info("✅ Pose estimation successful!")
            self.get_logger().info(f"Rotation vector:\n{rvec}")
            self.get_logger().info(f"Translation vector:\n{tvec}")

            R, _ = cv2.Rodrigues(rvec)
            self.get_logger().info(f"Rotation matrix:\n{R}")

            # Optional: Draw coordinate axes at estimated pose
            cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, 0.1)

        # Display the image
        cv2.imshow("Live Camera Feed", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LivePoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
