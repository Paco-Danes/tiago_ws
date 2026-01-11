import math
import rospy
import os
from .speech import tiago_say
from .base_motion import publish_cmd_vel
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    OpenAI = None
    _HAS_OPENAI = False

# Lazy singleton client
_client = None


def _get_openai_client():
    """
    Creates an OpenAI client if possible.
    Needs OPENAI_API_KEY in environment (standard SDK behavior).
    """
    global _client
    if not _HAS_OPENAI:
        return None

    if _client is not None:
        return _client

    # If key isn't set, just disable LLM and fall back to defaults.
    if not os.environ.get("OPENAI_API_KEY"):
        rospy.logwarn("OPENAI_API_KEY not set; falling back to canned lines.")
        return None

    try:
        _client = OpenAI()
        return _client
    except Exception as e:
        rospy.logwarn("Failed to init OpenAI client; falling back to canned lines. err=%s", e)
        return None


def _generate_short_line(kind: str, patient_name: str = "the patient") -> str:
    """
    kind: "joke" | "concern"
    Returns a short 1–2 line string. Falls back to a safe default on any failure.
    """
    client = _get_openai_client()

    fallback_joke = "Your refusal is hard to swallow... but not this pill! It goes down as smooth as champagne."
    fallback_concern = "Skipping it can be risky — your family and your doctor would want you protected today."

    if client is None:
        return fallback_joke if kind == "joke" else fallback_concern

    if kind == "joke":
        instructions = (
            "You write a gentle, non-mean joke for a healthcare robot. "
            "Goal: light humor to reduce anxiety after a medication refusal. "
            "Constraints: 1–2 short lines, friendly, no other text or preemble or final comments, directly the text."
        )
        user_prompt = (
            f"The patient ({patient_name}) just refused to take a pill. "
            "Generate a text the robot can say."
        )
    else:
        instructions = (
            "You write a brief, empathetic persuasion line for a healthcare robot. "
            "Goal: encourage medication adherence by referencing family concern and/or medical risk. "
            "Constraints: 1–2 short lines, caring tone, no other text or preemble or final comments, directly the text."
        )
        user_prompt = (
            f"The patient ({patient_name}) is refusing medication. "
            "Generate a text that encourages taking it, referencing family concern and/or health importance."
        )

    try:
        # Responses API (recommended for new projects) :contentReference[oaicite:3]{index=3}
        resp = client.responses.create(
            model="gpt-5-nano",
            reasoning={
                "effort": "minimal"
            },
            input=[
                {"role": "developer", "content": instructions},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (resp.output_text or "").strip()
        if not text:
            return fallback_joke if kind == "joke" else fallback_concern

        # Keep it truly short (defensive truncation)
        # (If the model returns more, keep only first 2 lines.)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = "\n".join(lines[:2]).strip()

        # Final sanity fallback
        if len(text) < 3:
            return fallback_joke if kind == "joke" else fallback_concern

        return text

    except Exception as e:
        rospy.logwarn("OpenAI call failed; falling back to canned lines. kind=%s err=%s", kind, e)
        return fallback_joke if kind == "joke" else fallback_concern


def humor_interaction(pub: rospy.Publisher, spin_angular_z: float) -> None:
    tiago_say("I see some hesitation... Time for a little humor!")
    # Replaces the fixed joke with an LLM-generated short joke
    joke = _generate_short_line(kind="joke", patient_name=rospy.get_param("~patient_name", "Francesco"))
    tiago_say(joke)

    tiago_say("Wheeee! Spinning for motivation!")

    rotate_in_place(pub, angle_rad=2.0 * math.pi, wz=spin_angular_z, odom_topic="/mobile_base_controller/odom")

    tiago_say("Alright Francesco — would you take it now?")


def family_concern() -> None:
    tiago_say("Are you sure? This is VERY important.")
    # Replaces the fixed concern line with an LLM-generated short concern
    concern = _generate_short_line(kind="concern", patient_name=rospy.get_param("~patient_name", "Francesco"))
    tiago_say(concern)
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
