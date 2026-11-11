import cv2
import sys

camera_idx = 0
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    camera_idx = int(sys.argv[1])

print(f"Testing camera index {camera_idx}...")
cap = cv2.VideoCapture(camera_idx)

if not cap.isOpened():
    print(f"Camera index {camera_idx} not detected!")
    # If index 0 failed and no explicit index was given, try index 1
    if len(sys.argv) == 1:
        print("Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("No webcam detected on index 0 or 1.")
            sys.exit(1)
        camera_idx = 1
    else:
        sys.exit(1)

print(f"Camera (index {camera_idx}) started successfully! Press 'q' to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read from camera.")
        break

    cv2.putText(
        frame,
        f"Sentinel AI Camera Test (Index: {camera_idx})",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sentinel AI - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera test completed.")