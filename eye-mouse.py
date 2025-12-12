import cv2
import mediapipe as mp
import pyautogui
import math

class EyeMouse:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # Critical for Iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Screen size
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Calibration / Sensitivity
        # Only use a central portion of the webcam feed to map to the full screen
        # This allows user to reach corners without twisting head too much
        self.frame_margin = 100 
        self.smooth_factor = 5 # Higher = smoother but more lag
        
        self.prev_x, self.prev_y = 0, 0
        
        # Blink thresholds
        self.blink_thresh = 0.012 # Eye closed distance related
        self.click_cooldown = 0
        self.click_delay = 10 # Frames between clicks

    def run(self):
        cap = cv2.VideoCapture(0)
        
        print("Starting Eye Mouse...")
        print("Move Head/Eyes to move cursor.")
        print("Wink Left -> Left Click")
        print("Wink Right -> Right Click")
        print("Press 'q' to quit.")

        prev_x, prev_y = 0, 0

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                continue

            # Flip and get dimensions
            img = cv2.flip(img, 1)
            h, w, c = img.shape
            
            # Draw Region of Interest (ROI) box for user reference
            # Mapping this box to the full screen
            cv2.rectangle(img, (self.frame_margin, self.frame_margin), 
                          (w - self.frame_margin, h - self.frame_margin), (0, 255, 0), 2)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # --- CURSOR CONTROL ---
                # Need a stable point. Landmarks 474-478 are Left Iris. 469-473 are Right Iris.
                # But simple Nose Tip (4) or mid-point between eyes is more stable for "Head Pointer"
                # Let's use landmark 4 (Nose Tip) + slight eye offset if needed.
                # For robust "Eye Mouse", tracking the PUPIL relative to EYE CORNERS is hard.
                # Most robust "Webcam Mouse" is Head Position mapped to Screen.
                
                nose_pt = landmarks[4]
                target_x = int(nose_pt.x * w)
                target_y = int(nose_pt.y * h)
                
                # Map coordinates from ROI to Screen
                # If x is frame_margin -> screen 0
                # If x is w - frame_margin -> screen width
                
                # Normalize 0 to 1 based on ROI
                norm_x = (target_x - self.frame_margin) / (w - 2 * self.frame_margin)
                norm_y = (target_y - self.frame_margin) / (h - 2 * self.frame_margin)
                
                # Clamp
                norm_x = max(0, min(1, norm_x))
                norm_y = max(0, min(1, norm_y))
                
                screen_x = norm_x * self.screen_w
                screen_y = norm_y * self.screen_h
                
                # Smoothing
                curr_x = prev_x + (screen_x - prev_x) / self.smooth_factor
                curr_y = prev_y + (screen_y - prev_y) / self.smooth_factor
                
                # Move Mouse
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y
                
                # --- CLICK DETECTION (BLINKS) ---
                # Left Eye: 159 (Top), 145 (Bottom)
                # Right Eye: 386 (Top), 374 (Bottom)
                
                left_dist = abs(landmarks[159].y - landmarks[145].y)
                right_dist = abs(landmarks[386].y - landmarks[374].y)
                
                # Visualize blink distance for debugging
                # cv2.putText(img, f"L: {left_dist:.4f} R: {right_dist:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
                if self.click_cooldown == 0:
                    # LEFT CLICK (Left Eye Closed, Right Open)
                    # Note: "Left Eye" is usually User's Left (Screen Left if flipped? No, Mesh is standardized)
                    # Actually, if we flipped image, left/right triggers might be visually swapped for user?
                    # Let's test standard mapping.
                    
                    if left_dist < self.blink_thresh and right_dist > self.blink_thresh:
                        pyautogui.click(button='left')
                        cv2.putText(img, "LEFT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        self.click_cooldown = self.click_delay
                        
                    # RIGHT CLICK (Right Eye Closed, Left Open)
                    elif right_dist < self.blink_thresh and left_dist > self.blink_thresh:
                        pyautogui.click(button='right')
                        cv2.putText(img, "RIGHT CLICK", (w-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        self.click_cooldown = self.click_delay
                else:
                    self.click_cooldown -= 1
                    
                # Visualize Cursor Point on Camera
                cv2.circle(img, (int(curr_x * w / self.screen_w), int(curr_y * h / self.screen_h)), 5, (255, 0, 255), cv2.FILLED)

            cv2.imshow("Eye Mouse", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EyeMouse()
    app.run()
