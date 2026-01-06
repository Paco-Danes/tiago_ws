import rospy
from geometry_msgs.msg import Twist
from .time_utils import wait_for_valid_sim_time

def stop_robot(pub: rospy.Publisher) -> None:
    stop = Twist()
    pub.publish(stop)
    rospy.sleep(0.1)
    pub.publish(stop)
    rospy.sleep(0.1)
    pub.publish(stop)

def publish_cmd_vel(pub: rospy.Publisher, linear_x: float, angular_z: float, duration_s: float, rate_hz: int = 20) -> None:
    #wait_for_valid_sim_time()

    rate = rospy.Rate(rate_hz)
    end_time = rospy.Time.now() + rospy.Duration.from_sec(duration_s)

    msg = Twist()
    msg.linear.x = float(linear_x)
    msg.angular.z = float(angular_z)

    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        pub.publish(msg)
        rate.sleep()

    stop_robot(pub)
