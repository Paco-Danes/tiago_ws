import math
import rospy
from .speech import tiago_say
from .base_motion import publish_cmd_vel

def humor_interaction(pub: rospy.Publisher, spin_angular_z: float) -> None:
    tiago_say("Hmm, I see some hesitation... Time for a little humor!")
    rospy.sleep(2.5)
    tiago_say("Your refusal is very hard to swallow... but not this pill! It goes down as smooth as champagne.")
    rospy.sleep(3.0)
    tiago_say("I’ll stop joking if you promise to keep your heart happy.")
    rospy.sleep(2.0)
    tiago_say("Wheeee! Spinning for motivation!")

    dur = (2.0 * math.pi) / abs(spin_angular_z) if abs(spin_angular_z) > 1e-3 else 6.28
    publish_cmd_vel(pub, 0.0, spin_angular_z, dur, rate_hz=20)

    tiago_say("Alright Francesco — would you take it now?")

def family_concern() -> None:
    tiago_say("Are you sure? This is VERY important.")
    rospy.sleep(2.0)
    tiago_say("Your family will be worried if you skip it. They care about you a lot.")
    rospy.sleep(2.0)
    tiago_say("Please, take it now.")
