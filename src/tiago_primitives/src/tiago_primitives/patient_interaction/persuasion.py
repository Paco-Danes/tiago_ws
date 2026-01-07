import math
import rospy
from .speech import tiago_say
from .base_motion import publish_cmd_vel
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

def humor_interaction(pub: rospy.Publisher, spin_angular_z: float) -> None:
    tiago_say("I see some hesitation... Time for a little humor!")
    rospy.sleep(1.0)
    tiago_say("Your refusal is very hard to swallow... but not this pill! It goes down as smooth as champagne.")
    tiago_say("Wheeee! Spinning for motivation!")

    rotate_in_place(pub, angle_rad=2.0*math.pi, wz=spin_angular_z, odom_topic="/mobile_base_controller/odom")

    tiago_say("Alright Francesco — would you take it now?")

def family_concern() -> None:
    tiago_say("Are you sure? This is VERY important.")
    rospy.sleep(2.0)
    tiago_say("Your family will be worried if you skip it. They care about you a lot.")
    rospy.sleep(2.0)
    tiago_say("Please, take it now.")

def yaw_from_odom_msg(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    quat = [q.x, q.y, q.z, q.w]
    _, _, yaw = euler_from_quaternion(quat)
    return yaw

def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def angle_diff(a: float, b: float) -> float:
    """Return wrapped (a-b) in [-pi, pi]."""
    return wrap_to_pi(a - b)

def rotate_in_place(pub: rospy.Publisher,
                    angle_rad: float = 2.0 * math.pi,
                    wz: float = 0.8,
                    yaw_tol_rad: float = math.radians(1.0),
                    odom_topic: str = "/odom") -> None:
    """
    Rotate by angle_rad (positive CCW, negative CW) using odometry yaw feedback.
    Stops within yaw_tol_rad.
    """
    # Get starting yaw
    msg0 = rospy.wait_for_message(odom_topic, Odometry, timeout=5.0)
    yaw0 = yaw_from_odom_msg(msg0)

    direction = 1.0 if angle_rad >= 0 else -1.0
    wz_cmd = direction * abs(wz)

    rate = rospy.Rate(30)
    twist = Twist()

    turned = 0.0
    last_yaw = yaw0

    while not rospy.is_shutdown():
        msg = rospy.wait_for_message(odom_topic, Odometry, timeout=1.0)
        yaw = yaw_from_odom_msg(msg)

        # incremental yaw change since last step (wrapped)
        dyaw = angle_diff(yaw, last_yaw)
        turned += dyaw
        last_yaw = yaw

        remaining = angle_rad - turned

        # Stop condition
        if abs(remaining) <= yaw_tol_rad:
            break

        # Slow down near the end so you don't overshoot
        speed = abs(wz_cmd)
        if abs(remaining) < math.radians(20):
            speed = max(0.15, speed * 0.4)

        twist.linear.x = 0.0
        twist.angular.z = direction * speed
        pub.publish(twist)
        rate.sleep()

    # hard stop
    twist.angular.z = 0.0
    pub.publish(twist)
    rospy.sleep(0.1)
    pub.publish(twist)
