import rospy
from geometry_msgs.msg import Twist

from .speech import tiago_say
from .time_utils import wait_for_valid_sim_time
from .base_motion import publish_cmd_vel
from .vision import recognize_person, scan_center_and_approach_person_yolo

from .arm import offer_pill_with_arm, play_motion
from .io_prompts import ask_yes_no
from .record import ensure_db, log_admin
from .persuasion import humor_interaction, family_concern


def run_interaction() -> None:
    """
    Runs one full patient interaction session.
    Assumes rospy.init_node(...) already called by entrypoint.
    """
    patient = rospy.get_param("~patient_name", "Francesco")
    cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/mobile_base_controller/cmd_vel")

    spin_angular_z = rospy.get_param("~spin_angular_z", 1.4)  # rad/s

    db_path = rospy.get_param("~db_path", "/home/user/exchange/tiago_ws/data/med_records.db")

    cam_index = int(rospy.get_param("~cam_index", 0))
    face_hold_seconds = float(rospy.get_param("~face_hold_seconds", 3.0))

    pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
    conn = ensure_db(db_path)

    wait_for_valid_sim_time()
    rospy.sleep(0.3)
    # Put head/robot in a neutral pose
    #play_motion("home", timeout_s=12.0)
    #rospy.sleep(0.2)

    # 1) Approach patient (Gazebo RGB segmentation + depth distance)
    rgb_topic = rospy.get_param("~rgb_topic", "/xtion/rgb/image_color")
    depth_topic = rospy.get_param("~depth_topic", "/xtion/depth_registered/image_raw")
    debug_view = bool(rospy.get_param("~debug_view_detection", False))

    dist0 = scan_center_and_approach_person_yolo(
        pub=pub,
        rgb_topic=rgb_topic,
        depth_topic=depth_topic,
        scan_wz=float(rospy.get_param("~scan_wz", 0.30)),
        scan_step_s=float(rospy.get_param("~scan_step_s", 0.35)),
        approach_fraction=float(rospy.get_param("~approach_fraction", 0.75)),
        linear_speed=float(rospy.get_param("~approach_linear_speed", 0.40)),
        debug_view=debug_view,
    )
    if dist0 is None:
        tiago_say("I couldn't find a person to approach. Stopping.")
        log_admin(conn, patient, "aborted_no_person_found", attempts=0, notes="yolo scan failed")
        return



    # 2) Face recognition simulation
    recognized = recognize_person(hold_seconds=2.0, cam_index=0)

    if recognized != "patient":
        tiago_say("No worries — we can try again later.")
        log_admin(conn, patient, "aborted_no_recognition", attempts=0, notes="user cancelled face window")
        return

    # 3) Greet + offer pill
    tiago_say(f"Oh, {patient}, I see you — you look great. Ready to take your medication?")
    offer_pill_with_arm()

    # 4) Up to 3 attempts
    attempt = 1
    while attempt <= 3 and not rospy.is_shutdown():
        if ask_yes_no("Pill taken?"):
            play_motion("home", timeout_s=12.0)
            tiago_say("Very good — motivated and disciplined.")
            tiago_say("I added this administration to your record! See you soon.")
            log_admin(conn, patient, "taken", attempts=attempt)
            return

        if attempt == 1:
            humor_interaction(pub, spin_angular_z)
        elif attempt == 2:
            family_concern()
        else:
            break

        attempt += 1

    # Final refusal
    play_motion("home", timeout_s=12.0)
    tiago_say("Okay — I will stop insisting for now.")
    tiago_say("I will record that you refused this dose.")
    log_admin(conn, patient, "refused", attempts=3)
