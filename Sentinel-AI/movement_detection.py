
from ultralytics import YOLO
import cv2
import math
import sys
import time
from pathlib import Path
from collections import defaultdict, deque

PROJECT_DIR = Path(__file__).resolve().parent

# ---------------- SETTINGS ----------------

VIDEO_PATH = str(PROJECT_DIR / "videos" / "sample_fixed.mp4")

PROCESS_EVERY = 3
IMAGE_SIZE = 640
CONFIDENCE = 0.35

WINDOW_SECONDS = 0.8
RUNNING_THRESHOLD = 1.8

# Higher value = boxes follow movement faster
SMOOTHING = 0.65

# ------------------------------------------

print("Loading Sentinel AI Movement Detection...")

model_path = PROJECT_DIR / "yolo11n.pt"
model = YOLO(str(model_path))

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print(f"ERROR: Could not open video: {VIDEO_PATH}")
    sys.exit(1)

fps = video.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 120 or fps != fps:
    fps = 25.0

frame_delay = max(1, int(1000 / fps))
window_frames = max(2, int(fps * WINDOW_SECONDS))

# Movement history for each person
person_history = defaultdict(lambda: deque())

# Smoothed bounding boxes for each person
smoothed_boxes = {}

frame_number = 0
last_boxes = []
running_count = 0

print("Starting video...")

while True:
    loop_start = time.perf_counter()
    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    # Run YOLO every few frames to reduce CPU load
    if frame_number % PROCESS_EVERY == 1:

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]
        last_boxes = []
        running_count = 0

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()

            for box, person_id, confidence in zip(
                boxes, ids, confidences
            ):

                x1, y1, x2, y2 = map(float, box)

                # Smooth box coordinates to reduce shaking
                if person_id in smoothed_boxes:

                    old_box = smoothed_boxes[person_id]

                    x1 = SMOOTHING * x1 + (1 - SMOOTHING) * old_box[0]
                    y1 = SMOOTHING * y1 + (1 - SMOOTHING) * old_box[1]
                    x2 = SMOOTHING * x2 + (1 - SMOOTHING) * old_box[2]
                    y2 = SMOOTHING * y2 + (1 - SMOOTHING) * old_box[3]

                smoothed_boxes[person_id] = (x1, y1, x2, y2)

                # Keep box inside video boundaries
                height, width = frame.shape[:2]

                x1 = max(0, min(width - 1, int(x1)))
                y1 = max(0, min(height - 1, int(y1)))
                x2 = max(0, min(width - 1, int(x2)))
                y2 = max(0, min(height - 1, int(y2)))

                box_height = max(1, y2 - y1)

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Save movement history
                history = person_history[person_id]

                history.append((
                    frame_number,
                    center_x,
                    center_y,
                    box_height
                ))

                while (
                    history and
                    frame_number - history[0][0] > window_frames
                ):
                    history.popleft()

                movement_label = "NORMAL"

                # Calculate movement over the time window
                if len(history) >= 2:

                    first = history[0]
                    last = history[-1]

                    elapsed = (last[0] - first[0]) / fps

                    if elapsed > 0:

                        distance = math.sqrt(
                            (last[1] - first[1]) ** 2 +
                            (last[2] - first[2]) ** 2
                        )

                        average_height = (
                            first[3] + last[3]
                        ) / 2

                        speed = distance / average_height / elapsed

                        if speed > RUNNING_THRESHOLD:
                            movement_label = "POSSIBLE RUNNING"
                            running_count += 1

                # Save drawing information
                last_boxes.append((
                    x1, y1, x2, y2,
                    person_id,
                    confidence,
                    movement_label
                ))

    # ---------------- DRAW BOXES ----------------

    for (
        x1, y1, x2, y2,
        person_id,
        confidence,
        movement_label
    ) in last_boxes:

        # Thin rectangular border
        box_color = (
            (0, 165, 255)
            if movement_label == "POSSIBLE RUNNING"
            else (0, 220, 0)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            1
        )

        # Small, readable label
        label = f"ID {person_id} | {confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_y = max(18, y1 - 5)

        # Small background behind text
        cv2.rectangle(
            frame,
            (x1, label_y - text_height - 5),
            (x1 + text_width + 6, label_y + 2),
            box_color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 3, label_y - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        # Movement warning
        if movement_label == "POSSIBLE RUNNING":

            cv2.putText(
                frame,
                movement_label,
                (x1, min(frame.shape[0] - 8, y2 + 18)),
                font,
                0.45,
                (0, 165, 255),
                1,
                cv2.LINE_AA
            )

    # ---------------- DISPLAY INFO ----------------

    cv2.putText(
        frame,
        f"People detected: {len(last_boxes)}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Possible running: {running_count}",
        (15, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 165, 255),
        1,
        cv2.LINE_AA
    )

    cv2.imshow("Sentinel AI - Movement Detection", frame)

    elapsed_ms = (time.perf_counter() - loop_start) * 1000
    wait_time = max(1, int(frame_delay - elapsed_ms))
    if cv2.waitKey(wait_time) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print("Sentinel AI stopped.")
