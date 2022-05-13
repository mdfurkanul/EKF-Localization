#!/usr/bin/env python

import rospy
import ekf
import sensors


rospy.init_node('EKF_node')
frequency = 250.00
Rate = rospy.Rate(frequency)

x = [
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0]
    ]

P = [0.0, 0.0, 0.0, 0.0, 0.0]
sigmaV = 0.1
sigmaOmega = 0.1

kalman = ekf.kalman(x, P)
readData = sensors.sensors()

if __name__ == "__main__":
    try:
        oldTime = rospy.Time().now().to_sec()
        while not rospy.is_shutdown():

            newTime = rospy.Time().now().to_sec()
            T = newTime - oldTime

            readData.read_sensors_data()
            error = kalman.estimate(readData)
            kalman.predict(T, sigmaV, sigmaOmega)
            kalman.publish_message(readData)

            oldTime = newTime
            Rate.sleep()

    except rospy.ROSInterruptException:
        pass
