import cv2
import time
import numpy as np
import mediapipe as mp
import math
import os

def set_volume_mac(volume):
    """
    Set system volume on macOS using osascript.
    volume: int between 0 and 100
    """
    cmd = f"osascript -e 'set volume output volume {volume}'"
    os.system(cmd)

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Open Webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    wCam, hCam = 640, 480
    cap.set(3, wCam)
    cap.set(4, hCam)

    pTime = 0
    vol = 0
    volBar = 400
    volPer = 0

    print("Volume Control Application Started. Press 'q' to exit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break

        # Convert to RGB for MediaPipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmarks positions
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                if len(lmList) != 0:
                    # Thumb tip (4) and Index finger tip (8)
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]
                    
                    # Midpoint
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Draw circles and line
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                    # Calculate length
                    length = math.hypot(x2 - x1, y2 - y1)
                    # print(length)

                    # Hand range: roughly 50 to 300
                    # Volume range: 0 to 100
                    
                    # Interpolate
                    vol = np.interp(length, [50, 250], [0, 100])
                    volBar = np.interp(length, [50, 250], [400, 150])
                    volPer = np.interp(length, [50, 250], [0, 100])
                    
                    # Set Volume
                    set_volume_mac(int(vol))

                    if length < 50:
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        # Draw Volume Bar
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Img", img)
        
        # Exit on logical close or 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
