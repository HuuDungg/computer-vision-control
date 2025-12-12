import cv2
import mediapipe as mp
import numpy as np
import math

class AirDraw:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Drawing state
        self.canvas = None
        self.prev_point = None
        self.drawing_color = (0, 0, 255)  # Initial color (Red in BGR)
        self.brush_thickness = 10
        self.eraser_thickness = 50

    def is_index_finger_only(self, landmarks):
        """
        Check if only the index finger is extended.
        Returns: strict_draw_mode (bool)
        """
        def is_extended(tip_idx, pip_idx, wrist_idx):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            wrist = landmarks[0]
            d_tip_wrist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            d_pip_wrist = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
            return d_tip_wrist > d_pip_wrist * 1.2

        index_ext = is_extended(8, 6, 0)
        middle_ext = is_extended(12, 10, 0)
        ring_ext = is_extended(16, 14, 0)
        pinky_ext = is_extended(20, 18, 0)
        
        return index_ext and not middle_ext and not ring_ext and not pinky_ext

    def is_open_hand(self, landmarks):
        """Check if all fingers are extended (Open Hand)."""
        # Simply check if tips are above PIPS (or further from wrist) for all 5 fingers
        # We can reuse the logic: sum of extended fingers >= 5
        def is_extended(tip_idx, pip_idx):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            wrist = landmarks[0]
            d_tip_wrist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            d_pip_wrist = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
            return d_tip_wrist > d_pip_wrist * 1.1

        # Thumb (4, 2), Index (8, 6), Middle (12, 10), Ring (16, 14), Pinky (20, 18)
        fingers_extended = [
            is_extended(4, 2),
            is_extended(8, 6),
            is_extended(12, 10),
            is_extended(16, 14),
            is_extended(20, 18)
        ]
        return all(fingers_extended)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        print("Starting Air Draw...")
        print("Gesture: Point index finger to draw.")
        print("Gesture: Wave open hand to change color.")
        print("Press 'q' to quit. Press 'c' to clear.")

        # Wave detection variables
        prev_wrist_x = None
        color_change_cooldown = 0
        
        # Colors (BGR)
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (255, 255, 255) # White
        ]
        color_index = 0

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip image for selfie view
            img = cv2.flip(img, 1)
            h, w, c = img.shape

            # Initialize canvas if needed
            if self.canvas is None:
                self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # Process hand
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            drawing_active = False
            cursor_pos = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Get coordinates
                    lm_list = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append((cx, cy))

                    if not lm_list:
                        continue

    def is_middle_finger_only(self, landmarks):
        """
        Check if only the middle finger is extended.
        Returns: True if middle extended and others (Index, Ring, Pinky) closed.
        """
        def is_extended(tip_idx, pip_idx, wrist_idx):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            wrist = landmarks[0]
            d_tip_wrist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            d_pip_wrist = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
            return d_tip_wrist > d_pip_wrist * 1.2

        index_ext = is_extended(8, 6, 0)
        middle_ext = is_extended(12, 10, 0)
        ring_ext = is_extended(16, 14, 0)
        pinky_ext = is_extended(20, 18, 0)
        
        return middle_ext and not index_ext and not ring_ext and not pinky_ext

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        print("Starting Air Draw...")
        print("Gesture: Point index finger to draw.")
        print("Gesture: Wave open hand to change color.")
        print("Gesture: Middle finger to clear canvas.")
        print("Press 'q' to quit. Press 'c' to clear.")

        # Wave detection variables
        prev_wrist_x = None
        color_change_cooldown = 0
        
        # Colors (BGR)
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (255, 255, 255) # White
        ]
        color_index = 0

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip image for selfie view
            img = cv2.flip(img, 1)
            h, w, c = img.shape

            # Initialize canvas if needed
            if self.canvas is None:
                self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # Process hand
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            drawing_active = False
            cursor_pos = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Get coordinates
                    lm_list = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append((cx, cy))

                    if not lm_list:
                        continue

                    # DRAWING MODE: Index Finger Only
                    if self.is_index_finger_only(hand_landmarks.landmark):
                        drawing_active = True
                        
                        # Get Index Finger Tip position
                        ix, iy = lm_list[8]
                        cursor_pos = (ix, iy)

                        # Draw Logic
                        if self.prev_point is None:
                            self.prev_point = (ix, iy)
                        else:
                            cv2.line(self.canvas, self.prev_point, (ix, iy), self.drawing_color, self.brush_thickness)
                            self.prev_point = (ix, iy)
                            
                    # COLOR CHANGE MODE: Open Hand + Wave
                    elif self.is_open_hand(hand_landmarks.landmark):
                        self.prev_point = None # Stop drawing
                        
                        # Wave detection
                        curr_wrist_x = lm_list[0][0]
                        if prev_wrist_x is not None:
                            speed = abs(curr_wrist_x - prev_wrist_x)
                            
                            if speed > 30 and color_change_cooldown == 0: # Threshold for wave speed
                                # Change Color
                                color_index = (color_index + 1) % len(colors)
                                self.drawing_color = colors[color_index]
                                color_change_cooldown = 15 # Wait frames
                                # Visual flash
                                cv2.putText(img, "Color Changed!", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, self.drawing_color, 3)

                        prev_wrist_x = curr_wrist_x
                    
                    # CLEAR MODE: Middle Finger Only
                    elif self.is_middle_finger_only(hand_landmarks.landmark):
                        self.prev_point = None
                        self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
                        cv2.putText(img, "Cleared!", (w//2-50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                    else:
                        # Neutral hand
                        self.prev_point = None
            
            else:
                self.prev_point = None
                prev_wrist_x = None

            # Decay cooldown
            if color_change_cooldown > 0:
                color_change_cooldown -= 1

            # Combine layers
            # Create a mask of the canvas
            img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
            
            # Mask out the canvas area from main image (make it black so we can add color)
            img = cv2.bitwise_and(img, img, mask=img_inv)
            
            # Add canvas to image
            img = cv2.add(img, self.canvas)

            # UI Feedback
            if cursor_pos:
                cv2.circle(img, cursor_pos, 15, self.drawing_color, cv2.FILLED)
                cv2.putText(img, "Drawing", (cursor_pos[0]+20, cursor_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.drawing_color, 2)
            
            # Show active color swatch
            cv2.rectangle(img, (20, 20), (70, 70), self.drawing_color, cv2.FILLED)
            cv2.putText(img, "Color", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(img, "'c': Clear | 'q': Quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Air Draw 3D", img)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AirDraw()
    app.run()
