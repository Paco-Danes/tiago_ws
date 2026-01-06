import rospy
from .speech import tiago_say

try:
    import actionlib
    from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
    _HAS_PLAY_MOTION = True
except Exception:
    actionlib = None
    PlayMotionAction = None
    PlayMotionGoal = None
    _HAS_PLAY_MOTION = False

def play_motion(motion_name: str, timeout_s: float = 12.0) -> bool:
    if not _HAS_PLAY_MOTION:
        rospy.logwarn("play_motion not available. Missing play_motion_msgs/actionlib.")
        return False

    client = actionlib.SimpleActionClient("/play_motion", PlayMotionAction)
    if not client.wait_for_server(rospy.Duration(3.0)):
        rospy.logwarn("/play_motion server not available.")
        return False

    motions_param = rospy.get_param("/play_motion/motions", {})
    motions = list(motions_param.keys())
    if motion_name not in motions:
        rospy.logwarn("Motion '%s' not found. Available: %s", motion_name, motions)
        return False

    rospy.loginfo("PLAY_MOTION -> sending '%s'", motion_name)
    goal = PlayMotionGoal()
    goal.motion_name = motion_name
    goal.skip_planning = True

    client.send_goal(goal)
    finished = client.wait_for_result(rospy.Duration(timeout_s))

    state = client.get_state()  # 3=SUCCEEDED, 4=ABORTED, 2=PREEMPTED, etc.
    result = client.get_result()
    rospy.loginfo("PLAY_MOTION -> finished=%s state=%s result=%s", finished, state, result)

    return finished and (state == 3)

def offer_pill_with_arm() -> None:
    tiago_say("Here you go — please take your pill from my hand !")

    ok1 = play_motion("offer", timeout_s=12.0)
    rospy.sleep(0.4)
    ok2 = play_motion("open_gripper", timeout_s=8.0)
    rospy.sleep(0.6)

    if not (ok1 or ok2):
        rospy.logwarn("Arm/gripper sequence did not succeed. Check PLAY_MOTION logs above.")
        return

    rospy.sleep(1.0)
