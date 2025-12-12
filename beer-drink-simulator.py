import cv2
import mediapipe as mp
import numpy as np
import math
import os
import glob

class BeerSimulator:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
        
        # Load Animation Frames from /animation folder
        self.frames = self.load_frames()
        self.num_frames = len(self.frames)
        
        # Game State
        self.current_frame_idx = 0 # 0 = Full, last = Empty
        self.is_drinking = False
        self.frame_timer = 0
        self.frames_per_level = 5 # How many game frames before advancing animation
        
        # Thresholds
        self.drink_dist_thresh = 0.18

    def load_frames(self):
        animation_dir = "animation"
        frames = []
        
        # Get sorted list of PNGs
        pattern = os.path.join(animation_dir, "*.png")
        files = sorted(glob.glob(pattern), key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        print(f"Loading {len(files)} animation frames from '{animation_dir}/'...")
        
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            # Add alpha if needed
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            frames.append(img)
        
        print(f"Loaded {len(frames)} frames.")
        return frames

    def overlay_image_alpha(self, img, img_overlay, pos):
        x, y = pos
        h, w, _ = img.shape
        h_o, w_o = img_overlay.shape[:2]
        
        # Scale to reasonable size
        target_w = 180
        scale = target_w / w_o
        new_h = int(h_o * scale)
        new_w = int(w_o * scale)
        overlay_resized = cv2.resize(img_overlay, (new_w, new_h))
        
        # Bounds
        y1 = y - new_h // 2
        y2 = y1 + new_h
        x1 = x - new_w // 2
        x2 = x1 + new_w

        # Clamp to image
        if y1 < 0: y1 = 0
        if x1 < 0: x1 = 0
        if y2 > h: y2 = h
        if x2 > w: x2 = w
        
        # Recalculate overlay slice
        oy1 = 0 if y - new_h//2 >= 0 else abs(y - new_h//2)
        ox1 = 0 if x - new_w//2 >= 0 else abs(x - new_w//2)
        oy2 = oy1 + (y2 - y1)
        ox2 = ox1 + (x2 - x1)

        if y2 - y1 <= 0 or x2 - x1 <= 0:
            return img

        overlay_crop = overlay_resized[oy1:oy2, ox1:ox2]

        # Alpha blend
        alpha_s = overlay_crop[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(3):
            img[y1:y2, x1:x2, c] = (alpha_s * overlay_crop[:, :, c] +
                                    alpha_l * img[y1:y2, x1:x2, c])
        return img

    def run(self):
        cap = cv2.VideoCapture(0)
        
        print("\n🍺 Beer Simulator Started!")
        print("Gesture: Hold your hand up to hold the glass.")
        print("Action: Bring glass close to your mouth to drink.")
        print("Keys: 'r' = Refill | 'a' = Auto Animation | 'q' = Quit")

        auto_anim = False

        while cap.isOpened():
            success, img = cap.read()
            if not success:
               continue

            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results_hands = self.hands.process(img_rgb)
            results_face = self.face_mesh.process(img_rgb)
            
            hand_pos = None
            mouth_pos = None
            
            # Detect Hand
            if results_hands.multi_hand_landmarks:
                lm = results_hands.multi_hand_landmarks[0].landmark
                # Index MCP (5)
                cx, cy = int(lm[5].x * w), int(lm[5].y * h)
                hand_pos = (cx, cy)
                
            # Detect Face
            if results_face.multi_face_landmarks:
                lm = results_face.multi_face_landmarks[0].landmark
                mx, my = int(lm[13].x * w), int(lm[13].y * h)
                mouth_pos = (mx, my)

            # --- LOGIC ---
            if auto_anim:
                # Auto-run animation from 1 to end
                self.frame_timer += 1
                if self.frame_timer >= self.frames_per_level:
                    self.frame_timer = 0
                    if self.current_frame_idx < self.num_frames - 1:
                        self.current_frame_idx += 1
                    else:
                        auto_anim = False # Stop at the end
                # Show in center if no hand
                if not hand_pos:
                    hand_pos = (w // 2, h // 2)
            
            elif hand_pos and mouth_pos:
                dist = math.hypot(hand_pos[0] - mouth_pos[0], hand_pos[1] - mouth_pos[1]) / w
                if dist < self.drink_dist_thresh:
                    self.is_drinking = True
                    self.frame_timer += 1
                    if self.frame_timer >= self.frames_per_level:
                        self.frame_timer = 0
                        if self.current_frame_idx < self.num_frames - 1:
                            self.current_frame_idx += 1
                else:
                    self.is_drinking = False
            
            # --- DRAW ---
            if hand_pos and self.frames:
                overlay_pos = (hand_pos[0], hand_pos[1] - 80)
                img = self.overlay_image_alpha(img, self.frames[self.current_frame_idx], overlay_pos)

            # --- UI ---
            progress = int(100 * (1 - self.current_frame_idx / max(1, self.num_frames - 1)))
            cv2.putText(img, f"Beer: {progress}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if self.current_frame_idx >= self.num_frames - 1:
                cv2.putText(img, "EMPTY! 'r' to refill", (w//2-120, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(img, "'r': Refill | 'a': Auto Anim | 'q': Quit", (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

            cv2.imshow("Beer Simulator", img)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.current_frame_idx = 0
                self.frame_timer = 0
            elif key == ord('a'):
                auto_anim = True
                self.current_frame_idx = 0
                self.frame_timer = 0

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = BeerSimulator()
    app.run()
