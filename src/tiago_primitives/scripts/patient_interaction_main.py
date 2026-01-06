#!/usr/bin/env python3
import rospy
from tiago_primitives.patient_interaction.main_logic import run_interaction

def main() -> None:
    rospy.init_node("patient_interaction_main", anonymous=False)
    run_interaction()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
