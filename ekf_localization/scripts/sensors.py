import numpy as np
import rospy
from scipy.linalg import block_diag
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist, Point, Quaternion

class sensors():

    def __init__(self):

        self.n = 0
        self.odom_cov = np.empty((4, 4), dtype=int)
        self.odomX = None
        self.odomY = None
        self.odomTheta = None
        self.odomVel = None
        self.imu_enable = False
        self.imu_cov = np.empty((2, 2), dtype=int)
        self.imuTheta = None
        self.imuOmega = None
        self.time_stamp = None

    def read_sensors_data(self):
        self.odom_data()

        if self.imu_enable:
            self.imu_data()

    def odom_data(self):
        try:
            msg = rospy.wait_for_message('/odom', Odometry, timeout=0.1)
            self.time_stamp = msg.header.stamp
            self.odomX = msg.pose.pose.position.x
            self.odomY = msg.pose.pose.position.y
            quat = msg.pose.pose.orientation
            (roll, pitch, yaw) = euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))
            self.odomTheta = yaw
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            self.odomVel = np.sqrt(vx**2 + vy**2)
            pose_covariance = msg.pose.covariance
            twist_covariance = msg.twist.covariance
            a = pose_covariance[35]
            b = twist_covariance[0] + twist_covariance[7]
            cov_noise = 0.01
            self.odom_cov = np.diag([pose_covariance[0]+cov_noise,
                                            pose_covariance[7]+cov_noise,
                                            a+cov_noise,
                                            b+cov_noise])
        except:
            pass
    def imu_data(self):
        try:
            msg = rospy.wait_for_message('/imu_data', Imu, timeout=0.1)
            quat = msg.orientation
            self.imuTheta = euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))[2]
            self.imuOmega = msg.angular_velocity.z
            self.imu_cov = np.diag((msg.orientation_covariance[8],
                                           msg.angular_velocity_covariance[8]))
        except:
            pass
