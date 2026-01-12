import math
import random
import rospy
from geometry_msgs.msg import Twist

from .speech import tiago_say
from .time_utils import wait_for_valid_sim_time
from .vision import recognize_person, scan_center_and_approach_person_yolo

from .arm import offer_pill_with_arm, play_motion
from .io_prompts import ask_yes_no
from .record import ensure_db, log_admin, get_bandit_stats, update_bandit_stats
from .persuasion import humor_interaction, family_concern
from .supervisor_dialogue import supervisor_interaction


_ARMS = ("joke", "concern")


def _pick_random_tie(candidates):
    return random.choice(list(candidates))


def _epsilon_greedy_choice(stats, epsilon: float) -> str:
    """
    stats: dict {arm: (n,s)}
    Rule enforced outside: if any n==0 -> pick it first.
    """
    eps = max(0.0, min(1.0, float(epsilon)))

    # Explore
    if random.random() < eps:
        return _pick_random_tie(_ARMS)

    # Exploit by empirical rate
    rates = {}
    for arm in _ARMS:
        n, s = stats[arm]
        rates[arm] = (float(s) / float(n)) if n > 0 else 0.0

    best = max(rates.values())
    best_arms = [a for a, r in rates.items() if abs(r - best) < 1e-12]
    return _pick_random_tie(best_arms)


def _ucb1_choice(stats, c: float) -> str:
    """
    UCB score: p_hat + c * sqrt( ln(t) / n )
    where t = total attempt-2 decisions for this patient (n_joke + n_concern).
    Rule enforced outside: if any n==0 -> pick it first.
    """
    c = float(c)

    t = sum(stats[a][0] for a in _ARMS)
    # t should be >= 2 if we got here (since both n>0), but keep it safe:
    t = max(1, int(t))

    scores = {}
    for arm in _ARMS:
        n, s = stats[arm]
        n = max(1, int(n))  # safe guard; should already be >0
        p_hat = float(s) / float(n)
        bonus = c * math.sqrt(math.log(float(t)) / float(n)) if t > 1 else 0.0
        scores[arm] = p_hat + bonus
        # let's log these values with rospy.loginfo
        rospy.loginfo("UCB_stats -> arm=%s |trials=%d | succ=%d | succ_rate=%.3f | UCB=%.3f", arm, n, s, p_hat, scores[arm])

    best = max(scores.values())
    best_arms = [a for a, sc in scores.items() if abs(sc - best) < 1e-12]
    chosen = _pick_random_tie(best_arms)
    rospy.loginfo("UCB selected strategy -> %s", chosen)
    return chosen


def get_best_second_strategy(conn, patient: str) -> str:
    """
    Decide which attempt-2 strategy to use for this patient: 'joke' or 'concern'.

    Policy:
      1) If an arm has n==0 -> play it (initial exploration), tie random if both 0.
      2) Else use selected bandit algorithm:
            - UCB (default): ~bandit_algo="ucb"
            - epsilon-greedy: ~bandit_algo="epsilon_greedy"
    """
    algo = str(rospy.get_param("~bandit_algo", "ucb")).strip().lower()
    stats = get_bandit_stats(conn, patient)

    # Rule 1: try untried arms first
    zero_arms = [a for a in _ARMS if stats[a][0] == 0]
    if zero_arms:
        return _pick_random_tie(zero_arms)

    # Rule 2: bandit policy
    if algo in ("epsilon", "eps", "epsilon_greedy", "e-greedy", "egreedy"):
        epsilon = float(rospy.get_param("~epsilon", 0.10))
        return _epsilon_greedy_choice(stats, epsilon=epsilon)

    # Default: UCB
    # If you want classic UCB1 constant, use c=sqrt(2). Here we expose ~ucb_c for HRI tuning.
    ucb_c = float(rospy.get_param("~ucb_c", math.sqrt(2.0)))
    return _ucb1_choice(stats, c=ucb_c)


def _run_strategy(pub, strategy: str, spin_angular_z: float) -> None:
    if strategy == "joke":
        humor_interaction(pub, spin_angular_z)
    else:
        family_concern()


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
    face_hold_seconds = float(rospy.get_param("~face_hold_seconds", 2.0))

    pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
    conn = ensure_db(db_path)

    wait_for_valid_sim_time()
    rospy.sleep(0.3)

    # 1) Approach person
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

    # 2) Face recognition
    recognized = recognize_person(hold_seconds=face_hold_seconds, cam_index=cam_index)

    if recognized not in ("patient", "supervisor"):
        tiago_say("No worries — we can try again later.")
        log_admin(conn, patient, "aborted_no_recognition", attempts=0, notes="user cancelled face window")
        return

    if recognized == "supervisor":
        tiago_say("Hello, supervisor. Awaiting further instructions.")
        supervisor_interaction(conn, patient_name=patient)
        return

    # 3) Patient interaction
    tiago_say(f"Oh, {patient}, I see you — you look great. Ready to take your medication?")
    offer_pill_with_arm()

    # --- Attempt 1: DIRECT (fixed; not part of bandit) ---
    if ask_yes_no("Pill taken?"):
        play_motion("home", timeout_s=12.0)
        tiago_say("Very good — motivated and disciplined.")
        tiago_say("I added this administration to your record! See you soon.")
        log_admin(conn, patient, "taken", attempts=1)
        return

    # --- Attempt 2: BANDIT (joke vs concern) ---
    chosen2 = get_best_second_strategy(conn, patient)
    algo = str(rospy.get_param("~bandit_algo", "ucb")).strip().lower()
    _run_strategy(pub, chosen2, spin_angular_z)

    taken_after_2 = ask_yes_no("Pill taken?")
    reward2 = 1 if taken_after_2 else 0

    # Update bandit ONLY for attempt-2 decision (per your requirement)
    update_bandit_stats(conn, patient, chosen2, reward=reward2)

    if taken_after_2:
        play_motion("home", timeout_s=12.0)
        tiago_say("Very good — thank you.")
        tiago_say("I added this administration to your record! See you soon.")
        log_admin(
            conn,
            patient,
            "taken",
            attempts=2,
            notes=f"attempt2={chosen2};bandit_algo={algo};reward2=1",
        )
        return

    # --- Attempt 3: fallback (the OTHER strategy), not bandit-scored ---
    other3 = "concern" if chosen2 == "joke" else "joke"
    tiago_say("I understand. I will try once more, gently.")
    _run_strategy(pub, other3, spin_angular_z)

    if ask_yes_no("Pill taken?"):
        play_motion("home", timeout_s=12.0)
        tiago_say("Thank you — that's a good decision.")
        log_admin(
            conn,
            patient,
            "taken",
            attempts=3,
            notes=f"attempt2={chosen2};attempt3={other3};bandit_algo={algo};reward2=0",
        )
        return

    # Final refusal
    play_motion("home", timeout_s=12.0)
    tiago_say("Okay — I will stop insisting for now.")
    tiago_say("I will record that you refused this dose.")
    log_admin(
        conn,
        patient,
        "refused",
        attempts=3,
        notes=f"attempt2={chosen2};attempt3={other3};bandit_algo={algo};reward2=0",
    )
