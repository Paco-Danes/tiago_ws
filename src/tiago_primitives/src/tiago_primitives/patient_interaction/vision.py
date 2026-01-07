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

import time
import threading
from typing import Optional, Tuple, List

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .speech import tiago_say
from .base_motion import publish_cmd_vel

from ultralytics import YOLO
import numpy as np

class YoloV8PersonDetector:
    def __init__(self, conf=0.3):
        self.model = YOLO("yolov8n.pt")
        self.conf = conf

    def detect(self, bgr_img):
        """
        bgr_img: numpy array (H,W,3) BGR
        returns: (x,y,w,h,conf) or None
        """
        # Ultralytics expects RGB
        rgb = bgr_img[:, :, ::-1]

        results = self.model(
            rgb,
            device="cpu",
            conf=self.conf,
            verbose=False
        )

        r = results[0]
        if r.boxes is None:
            return None

        boxes = r.boxes
        cls = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        # COCO: class 0 = person
        person_idxs = np.where(cls == 0)[0]
        if len(person_idxs) == 0:
            return None

        # pick highest confidence person
        i = person_idxs[np.argmax(confs[person_idxs])]
        x1, y1, x2, y2 = xyxy[i]

        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)
        return x, y, w, h, float(confs[i])

class GazeboRGBDListener:
    def __init__(self, rgb_topic: str, depth_topic: str):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.last_rgb_bgr = None
        self.last_depth_m = None

        self.rgb_sub = rospy.Subscriber(rgb_topic, Image, self._on_rgb, queue_size=1)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self._on_depth, queue_size=1)

    def _on_rgb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.last_rgb_bgr = bgr

    def _on_depth(self, msg: Image):
        # depth_registered is usually 32FC1 meters; sometimes 16UC1 mm
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


def letterbox(im: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    """Resize with padding to square (YOLO style). Returns resized image + scale + pad."""
    h, w = im.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape - nw
    pad_h = new_shape - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)
    return im_padded, scale, left, top

def depth_at_bbox(depth_m: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> Optional[float]:
    x, y, w, h = bbox_xywh
    H, W = depth_m.shape[:2]

    # sample a central region (more stable than edges)
    x0 = max(0, x + int(0.35 * w))
    x1 = min(W, x + int(0.65 * w))
    y0 = max(0, y + int(0.30 * h))
    y1 = min(H, y + int(0.85 * h))

    patch = depth_m[y0:y1, x0:x1]
    vals = patch[(patch > 0.0) & np.isfinite(patch)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def scan_center_and_approach_person_yolo(
    pub,
    rgb_topic: str,
    depth_topic: str,
    scan_wz: float = 0.30,
    scan_step_s: float = 0.35,
    max_scan_steps: int = 160,
    center_tol_px: int = 35,
    center_kp: float = 0.9,
    center_max_wz: float = 0.6,
    linear_speed: float = 0.35,
    approach_fraction: float = 0.80,
    debug_view: bool = False,
) -> Optional[float]:

    listener = GazeboRGBDListener(rgb_topic, depth_topic)
    if not listener.wait(timeout_s=6.0):
        rospy.logwarn("No RGB/Depth frames. Check topics: %s %s", rgb_topic, depth_topic)
        return None

    det = YoloV8PersonDetector(conf=rospy.get_param("~yolo_conf_thres", 0.3))


    tiago_say("Scanning for a person.")

    bbox = None
    dist0 = None

    # --- Scan
    for _ in range(int(max_scan_steps)):
        if rospy.is_shutdown():
            return None

        bgr, depth = listener.get()
        if bgr is not None and depth is not None:
            res = det.detect(bgr)
            if res is not None:
                x, y, w, h, conf = res
                d = depth_at_bbox(depth, (x, y, w, h))
                if d is not None:
                    bbox = (x, y, w, h, conf)
                    dist0 = d
                    break

        publish_cmd_vel(pub, 0.0, scan_wz, scan_step_s, rate_hz=20)

    if bbox is None or dist0 is None:
        tiago_say("I couldn't find anyone.")
        return None

    tiago_say("Person found. Centering.")

    # --- Center
    for _ in range(120):
        if rospy.is_shutdown():
            return None

        bgr, depth = listener.get()
        if bgr is None or depth is None:
            rospy.sleep(0.05)
            continue

        res = det.detect(bgr)
        if res is None:
            publish_cmd_vel(pub, 0.0, scan_wz, 0.2, rate_hz=20)
            continue

        x, y, w, h, conf = res
        d = depth_at_bbox(depth, (x, y, w, h))
        if d is None:
            publish_cmd_vel(pub, 0.0, scan_wz, 0.2, rate_hz=20)
            continue

        H, W = bgr.shape[:2]
        cx = x + w // 2
        err = cx - (W // 2)

        if debug_view:
            vis = bgr.copy()
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, f"person conf={conf:.2f} d={d:.2f}m err={err}px", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Gazebo YOLO Person Detection", vis)
            cv2.waitKey(1)

        if abs(err) <= int(center_tol_px):
            dist0 = d
            break

        # err>0 => person right => turn right => negative wz
        wz = -center_kp * (float(err) / float(W // 2))
        wz = max(-center_max_wz, min(center_max_wz, wz))
        publish_cmd_vel(pub, 0.0, wz, 0.18, rate_hz=20)

    if debug_view:
        cv2.destroyWindow("Gazebo YOLO Person Detection")

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
