import numpy as np
import rospy
from scipy.linalg import block_diag
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist, Point, Quaternion


def normalize_angle(angle):
    y = angle
    y = y % (2 * np.pi)
    if y > np.pi:
        y -= 2 * np.pi
    return y


class kalman():

    def __init__(self, x, P):

        self.not_first_time = False
        self.prev_x = np.matrix(x)
        self.prev_P = np.diag(P)
        self.time = 0
        self.est_x = self.prev_x
        self.est_P = self.prev_P
        self.pred_x = self.prev_x
        self.pred_P = self.prev_P

        self.C_both = np.matrix([[1, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 1, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1]])

        self.C_single = np.matrix([[1, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 1, 0]])

    def predict(self, T, sigmaV, sigmaOmega):

        self.pred_x[2, 0] = self.est_x[2, 0] + self.est_x[4, 0] * T
        self.pred_x[2, 0] = normalize_angle(self.pred_x[2, 0])
        self.pred_x[0, 0] = self.est_x[0, 0] + self.est_x[3, 0] * T * np.cos(self.pred_x[2, 0])
        self.pred_x[1, 0] = self.est_x[1, 0] + self.est_x[3, 0] * T * np.sin(self.pred_x[2, 0])
        self.pred_x[3, 0] = self.est_x[3, 0]
        self.pred_x[4, 0] = self.est_x[4, 0]



        ang = T*self.est_x[4, 0] + self.est_x[2, 0]
        ang = normalize_angle(ang)

        df_mat = np.matrix([[T*np.cos(ang), (-(T**2)*self.est_x[3, 0]*np.sin(ang))],
                        [T*np.sin(ang), (T**2)*self.est_x[3, 0]*np.cos(ang)],
                        [0, T],
                        [1, 0],
                        [0, 1]])
        df_Jacobian = np.matrix([[1, 0, -T*self.est_x[3, 0]*np.sin(ang), T*np.cos(ang), -(T**2)*self.est_x[3, 0]*np.sin(ang)],
                              [0, 1, T*self.est_x[3, 0]*np.cos(ang), T*np.sin(ang), (T**2)*self.est_x[3, 0]*np.cos(ang)],
                              [0, 0, 1, 0, T],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]])

        var_Q = np.matrix([[sigmaV**2, 0], [0, sigmaOmega**2]])
        Q = df_mat * var_Q * df_mat.T
        self.pred_P = df_Jacobian * self.est_P * df_Jacobian.T + Q

        self.prev_x = self.pred_x
        self.prev_P = self.pred_P

    def estimate(self, measureDataSens):

        if measureDataSens.imu_enable:
            z = np.matrix([[measureDataSens.odomX], [measureDataSens.odomY],
                           [measureDataSens.odomTheta], [measureDataSens.odomVel],
                           [measureDataSens.imuTheta], [measureDataSens.imuOmega]])
            C = self.C_both
            R = np.matrix(block_diag(measureDataSens.odom_cov, measureDataSens.imu_cov))
        else:

            z = np.matrix([[measureDataSens.odomX], [measureDataSens.odomY],
                           [measureDataSens.odomTheta], [measureDataSens.odomVel]])
            C = self.C_single
            R = np.matrix(block_diag(measureDataSens.odom_cov))

        S = C * self.prev_P * C.T + R
        self.K = self.prev_P * C.T * np.linalg.inv(S)

        error = z - (C * self.prev_x)
        error[2, 0] = normalize_angle(error[2, 0])

        # we don't estimate first time
        if self.not_first_time:
            self.est_x = self.prev_x + self.K * error
            self.est_x[2, 0] = normalize_angle(self.est_x[2, 0])
            mat = np.eye(5, dtype=int) - self.K * C
            self.est_P = mat * self.prev_P
        else:
            self.not_first_time = True

        self.get_vel()
        return error

    def callback_get_vel(self, msg):
        vx = msg.linear.x
        vy = msg.linear.y
        omega = msg.angular.z
        v = np.sqrt(vx**2 + vy**2)
        self.est_x[3, 0] = v
        self.est_x[4, 0] = omega

    def get_vel(self):
        rospy.Subscriber('/cmd_vel', Twist, self.callback_get_vel)

    def publish_message(self, caller_obj):

        Pub = rospy.Publisher('/odom_ekf', Odometry, queue_size=10)
        msg_odom = Odometry()
        x = self.est_x[0, 0]
        y = self.est_x[1, 0]
        z = 0
        quatern = quaternion_from_euler(0, 0, self.est_x[2, 0])
        msg_odom.header.stamp = caller_obj.time_stamp
        msg_odom.header.frame_id = 'odom'
        msg_odom.pose.pose.position = Point(x, y, z)
        msg_odom.pose.pose.orientation = Quaternion(*quatern)
        p = np.diag([self.est_P[0, 0], self.est_P[1, 1], 10000000, 10000000, 1000000, self.est_P[3, 3]])
        msg_odom.pose.covariance = tuple(p.ravel().tolist())

        Pub.publish(msg_odom)
