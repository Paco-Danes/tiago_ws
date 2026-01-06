#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist


def wait_for_valid_time():
    """If using sim time, wait until /clock has started."""
    if rospy.get_param("/use_sim_time", False):
        rospy.loginfo("use_sim_time is true: waiting for /clock...")
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() == 0.0:
            rospy.sleep(0.05)
        rospy.loginfo("Clock received. Current sim time: %.3f", rospy.Time.now().to_sec())


def main():
    rospy.init_node("move_forward_4s", anonymous=False)

    pub = rospy.Publisher("/mobile_base_controller/cmd_vel", Twist, queue_size=10)

    speed = rospy.get_param("~linear_x", 0.20)   # m/s
    duration = rospy.get_param("~duration", 4.0) # seconds
    rate_hz = rospy.get_param("~rate", 20)       # Hz

    wait_for_valid_time()

    rospy.loginfo("Moving forward: linear_x=%.3f m/s for %.2f s", speed, duration)

    twist = Twist()
    twist.linear.x = speed

    rate = rospy.Rate(rate_hz)
    start_time = rospy.Time.now()
    end_time = start_time + rospy.Duration.from_sec(duration)

    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        pub.publish(twist)
        rate.sleep()

    # Stop
    stop = Twist()
    pub.publish(stop)
    rospy.sleep(0.1)
    pub.publish(stop)

    rospy.loginfo("Done. Robot stopped.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
