#!/usr/bin/env python3
import cv2
import numpy as np
import os
import rospy

# Configuration
MODEL_FILE = "/home/user/exchange/tiago_ws/data/face_trainer.yml"

# Integer labels (LBPH requires ints)
LABEL_MAP = {
    "patient": 1,
    "supervisor": 2,
}

def get_face_roi(frame):
    """
    Extract ROI defined by the oval.
    Returns: gray ROI, (center, axes), bbox (x,y,w,h)
    """
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    axes = (int(w * 0.18), int(h * 0.28))

    x = center[0] - axes[0]
    y = center[1] - axes[1]
    rw = axes[0] * 2
    rh = axes[1] * 2

    x = max(0, x)
    y = max(0, y)
    rw = min(w - x, rw)
    rh = min(h - y, rh)

    roi = frame[y:y + rh, x:x + rw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return gray, (center, axes), (x, y, rw, rh)

def flash_feedback(win_name, frame, thickness=12, duration_ms=80):
    """Quick white border flash to confirm capture."""
    h, w = frame.shape[:2]
    flash = frame.copy()
    cv2.rectangle(flash, (0, 0), (w - 1, h - 1), (255, 255, 255), thickness)
    cv2.imshow(win_name, flash)
    cv2.waitKey(duration_ms)

def main():
    rospy.init_node("face_capture_trainer", anonymous=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    win = "Tiago Face Capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Plan: first 10 patient, then 10 supervisor
    capture_plan = [
        ("patient", 10),
        ("supervisor", 10),
    ]

    samples = []
    labels = []

    print("Face capture/training (2 people)")
    print("- Press 'c' to capture a photo")
    print("- Press 'n' to skip to next subject early")
    print("- Press 'q' to quit (cancel)")
    print("Align face inside the oval.")

    plan_index = 0
    subject_name, max_samples = capture_plan[plan_index]
    subject_label = LABEL_MAP[subject_name]
    count = 0

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            continue

        gray_roi, (center, axes), _bbox = get_face_roi(frame)

        # UI overlays
        display = frame.copy()
        cv2.ellipse(display, center, axes, 0, 0, 360, (0, 0, 255), 2)

        # Overall progress
        total_needed = sum(n for _, n in capture_plan)
        total_done = len(samples)

        cv2.putText(display, f"Total: {total_done}/{total_needed}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Subject progress
        pretty = subject_name.capitalize()
        cv2.putText(display, f"Subject: {pretty}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(display, f"Captured: {count}/{max_samples}", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        cv2.putText(display, "Keys: c=capture | n=next | q=quit", (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Capture cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

        if key == ord("c"):
            samples.append(gray_roi)
            labels.append(subject_label)
            count += 1
            print(f"[{subject_name}] Captured {count}/{max_samples}")
            flash_feedback(win, display)

            # Move to next subject automatically when done
            if count >= max_samples:
                plan_index += 1
                if plan_index >= len(capture_plan):
                    break  # done capturing all subjects

                subject_name, max_samples = capture_plan[plan_index]
                subject_label = LABEL_MAP[subject_name]
                count = 0
                print(f"--- Switching to {subject_name.upper()} ---")

        if key == ord("n"):
            # Skip to next subject early (optional)
            plan_index += 1
            if plan_index >= len(capture_plan):
                break
            subject_name, max_samples = capture_plan[plan_index]
            subject_label = LABEL_MAP[subject_name]
            count = 0
            print(f"--- Skipped. Switching to {subject_name.upper()} ---")

    cap.release()
    cv2.destroyAllWindows()

    # Sanity check: must have at least 1 sample per class
    have_patient = LABEL_MAP["patient"] in labels
    have_supervisor = LABEL_MAP["supervisor"] in labels
    if not (have_patient and have_supervisor):
        print("Not enough data: need samples for BOTH patient and supervisor to train.")
        return

    print("Training LBPH model on BOTH subjects...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if os.path.exists(MODEL_FILE):
        print(f"Warning: Overwriting existing {MODEL_FILE}")

    recognizer.train(samples, np.array(labels, dtype=np.int32))

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    recognizer.save(MODEL_FILE)
    print(f"Success! Model saved to {os.path.abspath(MODEL_FILE)}")

if __name__ == "__main__":
    main()
