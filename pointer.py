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
    
    # Initialize MediaPipe Face Mesh (for Blinks)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
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

    # Blink Logic
    blink_thresh = 0.012
    click_cooldown = 0
    click_delay = 15 # Frames
    
    print("Pointer App with Eye Clicks Started. Press 'q' to exit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break
        
        # Mirror image for natural feeling
        img = cv2.flip(img, 1) # 1 = horizontal flip
        h, w, c = img.shape

        # Convert to RGB for MediaPipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Process Hands (Cursor)
        results_hands = hands.process(imgRGB)
        
        # 2. Process Face (Clicks)
        results_face = face_mesh.process(imgRGB)

        # --- CURSOR MOVEMENT (HANDS) ---
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                if len(lmList) != 0:
                    x1, y1 = lmList[8][1], lmList[8][2] # Index Tip
                    
                    # Convert Coordinates
                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                    
                    # Smoothen Values
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    
                    # Move Mouse
                    try:
                        pyautogui.moveTo(clocX, clocY)
                    except pyautogui.FailSafeException:
                        pass
                    
                    plocX, plocY = clocX, clocY
                    
                    # Visual feedback for pointer
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

        # --- CLICK LOGIC (EYES/FACE) ---
        if results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0].landmark
            
            # Left Eye: 159 (Top), 145 (Bottom)
            # Right Eye: 386 (Top), 374 (Bottom)
            
            left_dist = abs(face_landmarks[159].y - face_landmarks[145].y)
            right_dist = abs(face_landmarks[386].y - face_landmarks[374].y)
            
            if click_cooldown == 0:
                # LEFT CLICK (Left Blink)
                if left_dist < blink_thresh and right_dist > blink_thresh:
                    pyautogui.click(button='left')
                    cv2.putText(img, "LEFT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    click_cooldown = click_delay
                    
                # RIGHT CLICK (Right Blink)
                elif right_dist < blink_thresh and left_dist > blink_thresh:
                    pyautogui.click(button='right')
                    cv2.putText(img, "RIGHT CLICK", (w-250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    click_cooldown = click_delay
        
        if click_cooldown > 0:
            click_cooldown -= 1

        # Draw Frame Reduction Box
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - 0.001) if (cTime - 0.001) > 0 else 0
        cv2.putText(img, str(int(fps)), (20, h - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Pointer App", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
