import rospy

def wait_for_valid_sim_time() -> None:
    """Avoid sim-time race: wait until /clock is non-zero when use_sim_time is true."""
    if rospy.get_param("/use_sim_time", False):
        rospy.loginfo("use_sim_time=true: waiting for /clock...")
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() == 0.0:
            rospy.sleep(0.05)
        rospy.loginfo("Clock OK. sim_time=%.3f", rospy.Time.now().to_sec())
