
import cv2
import time
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# =====================================
# SENTINEL AI - CONFIGURATION
# =====================================

PROJECT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Sentinel AI - People & Vehicle Tracking")
parser.add_argument(
    "--mode",
    choices=["video", "phone_wifi", "phone_usb", "webcam"],
    default="video",
    help="Video input mode (default: video)"
)
parser.add_argument(
    "--video",
    default=str(PROJECT_DIR / "videos" / "sample_fixed.mp4"),
    help="Path to video file"
)
parser.add_argument(
    "--stream",
    default="http://10.219.151.72:8080/video",
    help="Phone stream URL"
)
parser.add_argument(
    "--camera",
    type=int,
    default=0,
    help="Camera index for webcam / USB camera (default: 0)"
)
parser.add_argument(
    "--conf",
    type=float,
    default=0.4,
    help="YOLO detection confidence threshold (default: 0.4)"
)
parser.add_argument(
    "--imgsz",
    type=int,
    default=320,
    help="Inference image size (default: 320)"
)

args = parser.parse_args()

INPUT_MODE = args.mode
VIDEO_PATH = args.video
PHONE_STREAM_URL = args.stream
USB_CAMERA_INDEX = args.camera
CONFIDENCE = args.conf
IMAGE_SIZE = args.imgsz
PROCESS_EVERY = 4

# =====================================
# OBJECT CLASSES
# =====================================

CLASS_NAMES = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

PERSON_COLOR = (0, 255, 0)
VEHICLE_COLOR = (255, 150, 0)

# =====================================
# 1. OPEN VIDEO SOURCE
# =====================================

print(f"Starting Sentinel AI in '{INPUT_MODE}' mode...")

if INPUT_MODE == "video":
    source = VIDEO_PATH
elif INPUT_MODE == "phone_wifi":
    source = PHONE_STREAM_URL
elif INPUT_MODE in ("phone_usb", "webcam"):
    source = USB_CAMERA_INDEX
else:
    raise ValueError(f"Invalid input mode: {INPUT_MODE}")

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print(f"\n[ERROR] Cannot connect to video source: {source}")
    if INPUT_MODE == "phone_wifi":
        print("[HINT] Ensure the phone webcam server is running and your PC is on the same network.")
        print("[HINT] Alternatively run with: python sentinel.py --mode video")
    sys.exit(1)

# Reduce buffering for live camera modes
if INPUT_MODE != "video":
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# =====================================
# 2. LOAD YOLO MODEL
# =====================================

model_path = PROJECT_DIR / "yolo11n.pt"
if not model_path.exists():
    model_path = Path("yolo11n.pt")

print(f"Loading YOLO model from {model_path}...")
model = YOLO(str(model_path))
print("Model loaded successfully!")

# =====================================
# 3. VIDEO SETTINGS
# =====================================

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0 or fps > 120 or fps != fps:
    fps = 25.0

frame_delay = max(1, int(1000 / fps))

frame_count = 0
tracked_objects = []

previous_time = time.perf_counter()
smooth_fps = fps

# =====================================
# 4. MAIN DETECTION LOOP
# =====================================

while True:

    loop_start = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        print("Video ended or camera disconnected!")
        break

    frame_count += 1

    # ---------------------------------
    # RUN YOLO + BYTETRACK
    # ---------------------------------

    if frame_count % PROCESS_EVERY == 0:

        results = model.track(
            frame,
            persist=True,
            classes=[0, 2, 3, 5, 7],
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            tracker="bytetrack.yaml",
            verbose=False
        )

        tracked_objects = []

        for box in results[0].boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            class_id = int(box.cls[0])

            object_name = CLASS_NAMES.get(
                class_id, "Unknown"
            )

            object_id = None

            if box.id is not None:
                object_id = int(box.id[0])

            tracked_objects.append(
                (x1, y1, x2, y2, object_id, object_name)
            )

    # ---------------------------------
    # DRAW TRACKING BOXES
    # ---------------------------------

    person_count = 0
    vehicle_count = 0

    for x1, y1, x2, y2, object_id, object_name in tracked_objects:

        if object_name == "Person":
            color = PERSON_COLOR
            person_count += 1

        else:
            color = VEHICLE_COLOR
            vehicle_count += 1

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = (
            f"{object_name} ID:{object_id}"
            if object_id is not None
            else object_name
        )

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ---------------------------------
    # CALCULATE SMOOTH FPS
    # ---------------------------------

    current_time = time.perf_counter()
    instant_fps = 1.0 / max(current_time - previous_time, 0.0001)
    previous_time = current_time
    smooth_fps = 0.85 * smooth_fps + 0.15 * instant_fps

    # ---------------------------------
    # SENTINEL AI DISPLAY
    # ---------------------------------

    cv2.putText(
        frame,
        "SENTINEL AI",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Input: {INPUT_MODE}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"People: {person_count} | Vehicles: {vehicle_count}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {smooth_fps:.1f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Sentinel AI - People & Vehicle Tracking",
        frame
    )

    # ---------------------------------
    # PLAYBACK TIMING
    # ---------------------------------

    if INPUT_MODE == "video":

        processing_time = (
            time.perf_counter() - loop_start
        ) * 1000

        delay = max(
            1,
            int(frame_delay - processing_time)
        )

    else:
        delay = 1

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

# =====================================
# 5. RELEASE RESOURCES
# =====================================

cap.release()
cv2.destroyAllWindows()

print("Sentinel AI stopped.")
