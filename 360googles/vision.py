import cv2
import numpy as np
import argparse
from picamera2 import Picamera2

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Dual camera side-by-side viewer")
parser.add_argument("--rotate180", action="store_true", help="Rotate both cameras 180 degrees")
args = parser.parse_args()

ROTATE = args.rotate180

print(f"Rotate 180°: {ROTATE}")

# ----------------------------
# Helpers
# ----------------------------
def ensure_bgr(frame):
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return frame

def rotate_if_needed(frame):
    if ROTATE:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

# ----------------------------
# Camera Setup
# ----------------------------
picam0 = Picamera2(0)
picam1 = Picamera2(1)

config0 = picam0.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
config1 = picam1.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)

picam0.configure(config0)
picam1.configure(config1)

picam0.start()
picam1.start()

# ----------------------------
# Main Loop
# ----------------------------
while True:
    frame1 = ensure_bgr(picam0.capture_array())
    frame2 = ensure_bgr(picam1.capture_array())

    frame1 = rotate_if_needed(frame1)
    frame2 = rotate_if_needed(frame2)

    # Side-by-side (no stitching)
    combined = np.hstack((frame1, frame2))

    cv2.imshow("Side Vision", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam0.stop()
picam1.stop()
cv2.destroyAllWindows()