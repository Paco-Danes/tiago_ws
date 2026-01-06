import rospy
from sound_play.libsoundplay import SoundClient

_sound_client = SoundClient(blocking=True)

def tiago_say(text: str) -> None:
    rospy.loginfo("TIAGO: %s", text)
    _sound_client.say(text)
