"""
HSV Terrain Calibration Tool
Use this to tune terrain HSV ranges for your specific camera and lighting conditions.
Run this first, then update TERRAIN_PROFILES in terrain_classifier.py.

Author: Akshay Shailesh Raikar
"""

import cv2
import numpy as np

CAMERA_INDEX = 0

def nothing(x):
    pass

def main():
    print("[INFO] Terrain HSV Calibration Tool")
    print("[INFO] Point camera at a terrain surface and tune sliders")
    print("[INFO] Press S to print current HSV range | Q to quit")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        return

    cv2.namedWindow("HSV Calibration", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

    cv2.createTrackbar("H Low",  "HSV Calibration", 0,   179, nothing)
    cv2.createTrackbar("H High", "HSV Calibration", 179, 179, nothing)
    cv2.createTrackbar("S Low",  "HSV Calibration", 0,   255, nothing)
    cv2.createTrackbar("S High", "HSV Calibration", 255, 255, nothing)
    cv2.createTrackbar("V Low",  "HSV Calibration", 0,   255, nothing)
    cv2.createTrackbar("V High", "HSV Calibration", 255, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hl = cv2.getTrackbarPos("H Low",  "HSV Calibration")
        hh = cv2.getTrackbarPos("H High", "HSV Calibration")
        sl = cv2.getTrackbarPos("S Low",  "HSV Calibration")
        sh = cv2.getTrackbarPos("S High", "HSV Calibration")
        vl = cv2.getTrackbarPos("V Low",  "HSV Calibration")
        vh = cv2.getTrackbarPos("V High", "HSV Calibration")

        lower = np.array([hl, sl, vl])
        upper = np.array([hh, sh, vh])
        mask  = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        coverage = (np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])) * 100
        cv2.putText(frame, f"Coverage: {coverage:.1f}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Lower: [{hl}, {sl}, {vl}]", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Upper: [{hh}, {sh}, {vh}]", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "S: print values | Q: quit", (20, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

        cv2.imshow("HSV Calibration", frame)
        cv2.imshow("Mask", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"\n[CALIBRATION RESULT]")
            print(f'"lower": np.array([{hl}, {sl}, {vl}]),')
            print(f'"upper": np.array([{hh}, {sh}, {vh}]),')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
