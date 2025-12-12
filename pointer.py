import cv2
import time
import numpy as np
import mediapipe as mp
import pyautogui
import math

# Optimize pyautogui
pyautogui.FAILSAFE = False


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
    
    wCam, hCam = 640, 480
    cap.set(3, wCam)
    cap.set(4, hCam)
    
    # Screen size
    wScr, hScr = pyautogui.size()
    
    # Frame Reduction (to reach corners easily)
    frameR = 100 
    
    # Smoothing
    smoothening = 5
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    # Click Logic
    pinch_start_time = 0
    pinch_active = False
    
    # Logic:
    # - Pinch & Release quickly (< 0.4s) --> Left Click
    # - Pinch & Hold (> 0.4s) --> Right Click
    
    print("Mouse Pointer Application Started. Press 'q' to exit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break
        
        # Mirror image for natural feeling
        img = cv2.flip(img, 1)

        # Convert to RGB for MediaPipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                if len(lmList) != 0:
                    x1, y1 = lmList[8][1], lmList[8][2] # Index Tip
                    x2, y2 = lmList[4][1], lmList[4][2] # Thumb Tip
                    
                    # 1. Convert Coordinates
                    # Map range from frameR to wCam-frameR => 0 to wScr
                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                    
                    # 2. Smoothen Values
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    
                    # 3. Move Mouse
                    pyautogui.moveTo(clocX, clocY)
                    plocX, plocY = clocX, clocY
                    
                    # 4. Clicking Mode
                    length = math.hypot(x2 - x1, y2 - y1)
                    
                    # Visual feedback
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    
                    # --- NEW CLICK LOGIC ---
                    if length < 40: # Pinch IS active
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                        
                        if not pinch_active:
                            # Just started pinching
                            pinch_active = True
                            pinch_start_time = time.time()
                        
                        # Check for HOLD (Right Click)
                        if pinch_active and (time.time() - pinch_start_time > 0.4):
                            if pinch_start_time != 0: # Ensure we haven't handled this yet
                                print("Right Click (Hold)")
                                pyautogui.rightClick()
                                pinch_start_time = 0 # Mark as handled
                                
                    else: # Pinch is NOT active
                        if pinch_active:
                            # Just released pinch
                            # Check if it was a short tap and wasn't handled as Hold yet
                            if pinch_start_time != 0 and (time.time() - pinch_start_time < 0.4):
                                print("Left Click (Tap)")
                                pyautogui.click()
                            
                            pinch_active = False
                            pinch_start_time = 0
                    # -----------------------

        # Draw Frame Reduction Box
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - 0.001) if (cTime - 0.001) > 0 else 0
        
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
