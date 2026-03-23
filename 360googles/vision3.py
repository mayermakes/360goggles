import cv2
import numpy as np
import argparse
import subprocess
from picamera2 import Picamera2
#pinout for RPI composite video : https://community.element14.com/products/raspberry-pi/m/files/148385
# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Brain fusion side vision with display control")
parser.add_argument("--rotate180", action="store_true")
parser.add_argument("--fusion_width", type=int, default=50)
parser.add_argument("--fullscreen", action="store_true")
parser.add_argument("--display", type=str, default=None,
                    help="HDMI-1, HDMI-2, or BOTH")
args = parser.parse_args()

ROTATE = args.rotate180
FUSION_WIDTH = args.fusion_width
FULLSCREEN = args.fullscreen
DISPLAY = args.display

print(f"Fullscreen: {FULLSCREEN}")
print(f"Display: {DISPLAY}")

# ----------------------------
# Display Setup (Linux/X11)
# ----------------------------
def set_display(display):
    if display is None:
        return

    try:
        if display == "HDMI-1":
            subprocess.run(["xrandr", "--output", "HDMI-1", "--auto", "--primary"])
        elif display == "HDMI-2":
            subprocess.run(["xrandr", "--output", "HDMI-2", "--auto", "--primary"])
        elif display == "BOTH":
            subprocess.run(["xrandr", "--output", "HDMI-1", "--auto"])
            subprocess.run(["xrandr", "--output", "HDMI-2", "--auto", "--same-as", "HDMI-1"])
    except Exception as e:
        print("Display control failed:", e)

set_display(DISPLAY)

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

    combined[:, :w] = left

    for i in range(fusion_width):
        alpha = i / fusion_width
        combined[:, w - fusion_width + i] = (
            (1 - alpha) * left[:, w - fusion_width + i] +
            alpha * right[:, i]
        ).astype(np.uint8)

    combined[:, w:] = right[:, fusion_width:]

    return combined

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
# Window Setup
# ----------------------------
WINDOW_NAME = "Brain Fusion Vision"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

if FULLSCREEN:
    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

# ----------------------------
# Main Loop
# ----------------------------
while True:
    left = ensure_bgr(picam0.capture_array())
    right = ensure_bgr(picam1.capture_array())

    left = rotate_if_needed(left)
    right = rotate_if_needed(right)

    fused = brain_fusion(left, right, FUSION_WIDTH)

    cv2.imshow(WINDOW_NAME, fused)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam0.stop()
picam1.stop()
cv2.destroyAllWindows()