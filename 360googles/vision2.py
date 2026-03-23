import cv2
import numpy as np
import argparse
from picamera2 import Picamera2

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Side-mounted eyes with brain fusion")
parser.add_argument("--rotate180", action="store_true", help="Rotate both cameras 180 degrees")
parser.add_argument("--fusion_width", type=int, default=50, help="Width of fusion zone in pixels")
args = parser.parse_args()

ROTATE = args.rotate180
FUSION_WIDTH = args.fusion_width

print(f"Rotate 180°: {ROTATE}")
print(f"Fusion width: {FUSION_WIDTH} px")

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

def brain_fusion(left, right, fusion_width):
    h, w, c = left.shape
    combined_width = w*2 - fusion_width
    combined = np.zeros((h, combined_width, c), dtype=np.uint8)

    # Copy left image
    combined[:, :w] = left

    # Blend the inner edges
    for i in range(fusion_width):
        alpha = i / fusion_width
        combined[:, w - fusion_width + i] = (
            (1 - alpha) * left[:, w - fusion_width + i] +
            alpha * right[:, i]
        ).astype(np.uint8)

    # Copy remaining right image
    combined[:, w:] = right[:, fusion_width:]

    return combined

# ----------------------------
# Camera Setup
# ----------------------------
picam0 = Picamera2(0)
picam1 = Picamera2(1)

config0 = picam0.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
config1 = picam1.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})

picam0.configure(config0)
picam1.configure(config1)

picam0.start()
picam1.start()

# ----------------------------
# Main Loop
# ----------------------------
while True:
    left = ensure_bgr(picam0.capture_array())
    right = ensure_bgr(picam1.capture_array())

    left = rotate_if_needed(left)
    right = rotate_if_needed(right)

    fused = brain_fusion(left, right, FUSION_WIDTH)

    cv2.imshow("Brain Fusion Side Vision", fused)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam0.stop()
picam1.stop()
cv2.destroyAllWindows()