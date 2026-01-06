import time
import rospy
from .speech import tiago_say
from .base_motion import publish_cmd_vel
import cv2
import numpy as np

def simulate_face_recognition(patient_name: str, hold_seconds: float, cam_index: int) -> bool:
    tiago_say("Hello there! Please look at me!")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        rospy.logwarn("Cannot open camera index %d. Simulating with a blank window.", cam_index)
        cap = None

    win = "CareTIAGO Face Check"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except Exception as e:
        rospy.logwarn("Cannot create OpenCV window (%s). Simulating recognition with a sleep.", e)
        time.sleep(hold_seconds)
        return True

    start_wall = time.time()
    recognized = False
    blank = None

    while True:
        elapsed = time.time() - start_wall

        frame = None
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                frame = None

        if frame is None:
            if blank is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = blank.copy()

        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        axes = (int(w * 0.18), int(h * 0.28))

        color = (0, 0, 255)
        text = "Align face inside oval"
        if elapsed >= hold_seconds:
            color = (0, 255, 0)
            text = f"Recognized: {patient_name}"
            recognized = True

        cv2.ellipse(frame, center, axes, 0, 0, 360, color, 3)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to cancel", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            recognized = False
            break

        if recognized:
            cv2.waitKey(600)
            break

    if cap is not None:
        cap.release()
    cv2.destroyWindow(win)
    return recognized

# --- Gazebo approach using RGB color segmentation + depth distance -------------
import time
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .speech import tiago_say
from .base_motion import publish_cmd_vel


class GazeboRGBDListener:
    def __init__(self, rgb_topic: str, depth_topic: str):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.last_rgb_bgr = None      # np.ndarray HxWx3 uint8
        self.last_depth_m = None      # np.ndarray HxW float32 meters

        self.rgb_sub = rospy.Subscriber(rgb_topic, Image, self._on_rgb, queue_size=1)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self._on_depth, queue_size=1)

    def _on_rgb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.last_rgb_bgr = bgr

    def _on_depth(self, msg: Image):
        # Gazebo often publishes 32FC1 (meters). Some setups publish 16UC1 (mm).
        if msg.encoding in ("32FC1", "32FC"):
            d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1").astype(np.float32)
            depth_m = d
        elif msg.encoding in ("16UC1", "16UC"):
            d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1").astype(np.float32)
            depth_m = d / 1000.0
        else:
            d = self.bridge.imgmsg_to_cv2(msg)
            if d.dtype == np.uint16:
                depth_m = d.astype(np.float32) / 1000.0
            else:
                depth_m = d.astype(np.float32)

        depth_m = np.where(np.isfinite(depth_m), depth_m, 0.0)
        depth_m = np.where(depth_m > 0.0, depth_m, 0.0)

        with self.lock:
            self.last_depth_m = depth_m

    def wait(self, timeout_s: float = 5.0) -> bool:
        t0 = time.time()
        while not rospy.is_shutdown() and (time.time() - t0) < timeout_s:
            with self.lock:
                ok = (self.last_rgb_bgr is not None) and (self.last_depth_m is not None)
            if ok:
                return True
            rospy.sleep(0.05)
        return False

    def get(self):
        with self.lock:
            rgb = None if self.last_rgb_bgr is None else self.last_rgb_bgr.copy()
            depth = None if self.last_depth_m is None else self.last_depth_m.copy()
        return rgb, depth


def _hsv_mask_for_person(hsv: np.ndarray) -> np.ndarray:
    """
    Tuned for Gazebo human:
    - Strong blue jeans
    - Dark / low-sat shirt (red-brown-green plaid)
    """
    H, S, V = cv2.split(hsv)

    # --- Blue jeans
    jeans = cv2.inRange(hsv, (80, 25, 15), (150, 255, 255))

    # Combine: jeans
    mask = jeans

    # Morphology cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def detect_person_blob_rgb(
    bgr: np.ndarray,
    min_area_px: int = 1500,
    min_aspect: float = 1.3,    # tall-ish
    min_height_frac: float = 0.20,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns bounding box (x, y, w, h) of best person candidate, or None.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = _hsv_mask_for_person(hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    H, W = bgr.shape[:2]

    best = None
    best_score = -1.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(min_area_px):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        
        # jeans lower half check
        roi = hsv[y + h//2 : y + h, x : x + w]
        if roi.size == 0:
            continue

        jeans_roi = cv2.inRange(roi, (80, 25, 15), (150, 255, 255))
        if cv2.countNonZero(jeans_roi) < 0.10 * roi.shape[0] * roi.shape[1]:
            continue

        # Expand bbox upward: torso is above jeans
        y2 = max(0, y - int(1.1 * h))     # extend up by ~1.1x jeans height
        h2 = h + y - y2                 # keep bottom fixed
        x2 = max(0, x - int(0.35 * w))    # widen a bit
        w2 = min(W - x2, int(1.7 * w))

        x, y, w, h = x2, y2, w2, h2
        # Basic shape filters
        if h < int(min_height_frac * H):
            continue
        aspect = float(h) / max(1.0, float(w))
        if aspect < float(min_aspect):
            continue

        # Score: prefer larger and more centered blobs
        cx = x + w / 2.0
        center_score = 1.0 - abs(cx - (W / 2.0)) / (W / 2.0)
        score = area * 0.7 + (center_score * 1000.0) + (h * 2.0)

        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    return best


def depth_at_bbox(depth_m: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[float]:
    x, y, w, h = bbox
    H, W = depth_m.shape[:2]
    x0 = max(0, x + int(0.35 * w))
    x1 = min(W, x + int(0.65 * w))
    y0 = max(0, y + int(0.30 * h))
    y1 = min(H, y + int(0.85 * h))

    patch = depth_m[y0:y1, x0:x1]
    vals = patch[(patch > 0.0) & np.isfinite(patch)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def scan_center_and_approach_person_rgb(
    pub,
    rgb_topic: str,
    depth_topic: str,
    scan_wz: float = 0.30,
    scan_step_s: float = 0.35,
    max_scan_steps: int = 160,
    center_tol_px: int = 40,
    center_kp: float = 0.9,
    center_max_wz: float = 0.6,
    linear_speed: float = 0.35,
    approach_fraction: float = 0.80,
    debug_view: bool = False,
) -> Optional[float]:
    """
    Rotate, detect person by RGB color segmentation, center it, then drive forward 80% of depth distance.
    Returns initial distance (m) or None.
    """
    listener = GazeboRGBDListener(rgb_topic, depth_topic)
    if not listener.wait(timeout_s=6.0):
        rospy.logwarn("No RGB/Depth frames. Check topics: %s and %s", rgb_topic, depth_topic)
        return None

    tiago_say("Scanning for a person.")

    bbox = None
    dist0 = None

    # --- Scan phase
    for _ in range(int(max_scan_steps)):
        if rospy.is_shutdown():
            return None

        bgr, depth = listener.get()
        if bgr is not None and depth is not None:
            bb = detect_person_blob_rgb(bgr)
            if bb is not None:
                d = depth_at_bbox(depth, bb)
                if d is not None:
                    bbox = bb
                    dist0 = d
                    break

        publish_cmd_vel(pub, 0.0, scan_wz, scan_step_s, rate_hz=20)

    if bbox is None or dist0 is None:
        tiago_say("I couldn't find anyone.")
        return None

    tiago_say("Person found. Centering.")

    # --- Center phase
    for _ in range(120):
        if rospy.is_shutdown():
            return None

        bgr, depth = listener.get()
        if bgr is None or depth is None:
            rospy.sleep(0.05)
            continue

        bb = detect_person_blob_rgb(bgr)
        if bb is None:
            # Lost: small scan nudge
            publish_cmd_vel(pub, 0.0, scan_wz, 0.2, rate_hz=20)
            continue

        d = depth_at_bbox(depth, bb)
        if d is None:
            publish_cmd_vel(pub, 0.0, scan_wz, 0.2, rate_hz=20)
            continue

        x, y, w, h = bb
        H, W = bgr.shape[:2]
        cx = x + w // 2
        err = cx - (W // 2)

        if debug_view:
            vis = bgr.copy()
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(vis, (cx, y + h // 2), 6, (0, 255, 0), -1)
            cv2.putText(vis, f"err={err}px  d={d:.2f}m", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Gazebo Person Detection", vis)
            cv2.waitKey(1)

        if abs(err) <= int(center_tol_px):
            dist0 = d  # update distance right before approach
            break

        # +wz is CCW (left). If blob is right (err>0), we need turn right => negative wz.
        wz = -center_kp * (float(err) / float(W // 2))
        wz = max(-center_max_wz, min(center_max_wz, wz))
        publish_cmd_vel(pub, 0.0, wz, 0.18, rate_hz=20)

    if debug_view:
        cv2.destroyWindow("Gazebo Person Detection")

    tiago_say("Approaching.")

    travel = float(approach_fraction) * float(dist0)
    travel = max(0.0, min(travel, 4.0))
    if travel < 0.05:
        tiago_say("Already close enough.")
        return dist0

    dur = travel / max(0.05, float(linear_speed))
    publish_cmd_vel(pub, float(linear_speed), 0.0, float(dur), rate_hz=20)
    tiago_say("I have arrived.")
    return dist0
