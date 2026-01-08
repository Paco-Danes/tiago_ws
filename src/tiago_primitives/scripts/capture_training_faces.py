#!/usr/bin/env python3
import cv2
import numpy as np
import os
import rospy

# Configuration
MODEL_FILE = "/home/user/exchange/tiago_ws/data/face_trainer.yml"
# Map names to Integer IDs (LBPH requires integer labels)
LABEL_MAP = {
    "patient": 1,
    "supervisor": 2
}

def get_face_roi(frame):
    """
    Extracts the Region of Interest (ROI) defined by the oval.
    Returns the cropped gray image and the rectangle coordinates.
    """
    h, w = frame.shape[:2]
    # Define oval parameters (same as your vision.py)
    center = (w // 2, h // 2)
    axes = (int(w * 0.18), int(h * 0.28))
    
    # Calculate bounding box of the oval for cropping
    x = center[0] - axes[0]
    y = center[1] - axes[1]
    rw = axes[0] * 2
    rh = axes[1] * 2
    
    # Ensure bounds are valid
    x = max(0, x); y = max(0, y)
    rw = min(w - x, rw); rh = min(h - y, rh)
    
    # Crop and convert to grayscale
    roi = frame[y:y+rh, x:x+rw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return gray, (center, axes)

def main():
    rospy.init_node('face_capture_trainer', anonymous=True)
    
    # 1. Select Role
    print("Select user to capture:")
    print("1: Patient")
    print("2: Supervisor")
    choice = input("Enter number: ").strip()
    
    if choice == '1':
        label_id = LABEL_MAP["patient"]
        person_name = "Patient"
    elif choice == '2':
        label_id = LABEL_MAP["supervisor"]
        person_name = "Supervisor"
    else:
        print("Invalid choice.")
        return

    cap = cv2.VideoCapture(0) # Index 0 for default webcam
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    samples = []
    labels = []
    count = 0
    max_samples = 10
    
    win = "Tiago Face Capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f"Align {person_name}'s face in the oval.")
    print("Press 'c' to CAPTURE a photo.")
    print("Press 'q' to QUIT.")

    while count < max_samples and not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret: continue

        gray_roi, (center, axes) = get_face_roi(frame)
        
        # UI Overlays
        display_frame = frame.copy()
        cv2.ellipse(display_frame, center, axes, 0, 0, 360, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Captured: {count}/{max_samples}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Subject: {person_name}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        cv2.imshow(win, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            # Save the captured face
            samples.append(gray_roi)
            labels.append(label_id)
            count += 1
            print(f"Captured image {count}")
            # Visual feedback
            cv2.rectangle(display_frame, (0,0), (frame.shape[1], frame.shape[0]), (255,255,255), 10)
            cv2.imshow(win, display_frame)
            cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()

    if count == max_samples:
        print("Training LBPH model...")
        
        # Initialize LBPH Face Recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # If a model exists, we could load it and update it, 
        # but standard LBPH `train` overwrites. 
        # For simplicity, we retrain new data. 
        # (To support multiple people, you'd usually capture ALL images first then train once).
        
        if os.path.exists(MODEL_FILE):
             print(f"Warning: Overwriting existing {MODEL_FILE}")
        
        recognizer.train(samples, np.array(labels))
        recognizer.save(MODEL_FILE)
        print(f"Success! Model saved to {os.path.abspath(MODEL_FILE)}")
    else:
        print("Capture cancelled.")

if __name__ == "__main__":
    main()