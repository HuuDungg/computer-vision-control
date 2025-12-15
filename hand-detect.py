import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe modules
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Create hand detection object (max 2 hands)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Create pose detection object (for shoulder and elbow)
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Function to calculate angle between 3 points (returns angle at middle point)
def calculate_angle(a, b, c):
    """
    Calculate angle at point b formed by 3 points a, b, c
    a: first point (shoulder)
    b: middle point (elbow)  
    c: end point (wrist)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Vector from b to a and b to c
    ba = a - b
    bc = c - b
    
    # Calculate angle using cosine formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

# Rep counting variables and states
left_rep_count = 0
right_rep_count = 0
left_stage = "down"  # "down" = arm down, "ready" = at 90°, "up" = lifted
right_stage = "down"

# Angle thresholds for shoulder exercise (lateral raise)
ANGLE_READY_MIN = 85    # Ready position min angle
ANGLE_READY_MAX = 95    # Ready position max angle (85-95° = ready)
ANGLE_UP = 145          # Top position - arm raised to 145°+ = 1 rep
ANGLE_DOWN = 60         # Arm down position

# Open camera
cap = cv2.VideoCapture(0)

print("=" * 50)
print("    GYM TRAINER - SHOULDER EXERCISE REP COUNTER")
print("=" * 50)
print("Instructions:")
print("- Stand in front of camera, show shoulders and arms")
print("- Start with arm at 90° angle (ready position)")
print("- Lift weight up to ~150° angle to complete 1 rep")
print("- Press 'R' to reset rep count")
print("- Press 'Q' to quit")
print("=" * 50)

while True:
    success, frame = cap.read()
    if not success:
        print("Cannot open camera.")
        break

    # Flip image for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand detection
    results = hands.process(img_rgb)

    # Process pose detection (shoulder, elbow)
    pose_results = pose.process(img_rgb)

    # Variables to store angle and form info
    left_angle = None
    right_angle = None
    left_form_feedback = ""
    right_form_feedback = ""

    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

    # Draw shoulder and elbow from pose detection
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        
        point_coords = {}
        
        # Points to track
        arm_points = {
            11: ("L_SHOULDER", (0, 255, 255)),    # Left shoulder - Yellow
            12: ("R_SHOULDER", (0, 255, 255)),    # Right shoulder - Yellow
            13: ("L_ELBOW", (255, 0, 0)),         # Left elbow - Blue
            14: ("R_ELBOW", (255, 0, 0)),         # Right elbow - Blue
            15: ("L_WRIST", (0, 165, 255)),       # Left wrist - Orange
            16: ("R_WRIST", (0, 165, 255))        # Right wrist - Orange
        }
        
        # Get coordinates for each point
        for idx in arm_points.keys():
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                point_coords[idx] = (cx, cy)
        
        # Initialize colors
        angle_color_left = (255, 255, 255)
        angle_color_right = (255, 255, 255)
        
        # ========== LEFT ARM ANGLE & REP COUNTING ==========
        if all(idx in point_coords for idx in [11, 13, 15]):
            left_angle = calculate_angle(
                point_coords[11],  # Shoulder
                point_coords[13],  # Elbow
                point_coords[15]   # Wrist
            )
            
            # State machine for rep counting
            # down -> ready (at 90°) -> up (at 150°) -> counts as 1 rep -> back to down
            
            if left_angle < ANGLE_DOWN:
                left_stage = "down"
                left_form_feedback = f"LEFT: Arm down ({int(left_angle)})"
                angle_color_left = (128, 128, 128)  # Gray
                
            elif ANGLE_READY_MIN <= left_angle <= ANGLE_READY_MAX:
                if left_stage != "ready":  # Allow transition from both 'down' and 'up'
                    left_stage = "ready"
                left_form_feedback = f"LEFT: READY ({int(left_angle)}) - LIFT UP!"
                angle_color_left = (0, 255, 255)  # Yellow = ready
                
            elif left_angle >= ANGLE_UP:
                if left_stage == "ready":
                    left_rep_count += 1
                    print(f"[LEFT ARM] Rep: {left_rep_count}")
                    left_stage = "up"
                left_form_feedback = f"LEFT: REP COMPLETE!"
                angle_color_left = (0, 255, 0)  # Green = completed
                
            else:
                # In between states
                if left_stage == "ready":
                    left_form_feedback = f"LEFT: Keep lifting ({int(left_angle)})"
                    angle_color_left = (0, 165, 255)  # Orange
                elif left_stage == "up":
                    left_form_feedback = f"LEFT: Lower arm ({int(left_angle)})"
                    angle_color_left = (0, 255, 0)
                else:
                    left_form_feedback = f"LEFT: Go to 90 ({int(left_angle)})"
                    angle_color_left = (128, 128, 128)
        
        # ========== RIGHT ARM ANGLE & REP COUNTING ==========
        if all(idx in point_coords for idx in [12, 14, 16]):
            right_angle = calculate_angle(
                point_coords[12],  # Shoulder
                point_coords[14],  # Elbow
                point_coords[16]   # Wrist
            )
            
            # State machine for rep counting
            if right_angle < ANGLE_DOWN:
                right_stage = "down"
                right_form_feedback = f"RIGHT: Arm down ({int(right_angle)})"
                angle_color_right = (128, 128, 128)  # Gray
                
            elif ANGLE_READY_MIN <= right_angle <= ANGLE_READY_MAX:
                if right_stage != "ready":  # Allow transition from both 'down' and 'up'
                    right_stage = "ready"
                right_form_feedback = f"RIGHT: READY ({int(right_angle)}) - LIFT UP!"
                angle_color_right = (0, 255, 255)  # Yellow = ready
                
            elif right_angle >= ANGLE_UP:
                if right_stage == "ready":
                    right_rep_count += 1
                    print(f"[RIGHT ARM] Rep: {right_rep_count}")
                    right_stage = "up"
                right_form_feedback = f"RIGHT: REP COMPLETE!"
                angle_color_right = (0, 255, 0)  # Green = completed
                
            else:
                # In between states
                if right_stage == "ready":
                    right_form_feedback = f"RIGHT: Keep lifting ({int(right_angle)})"
                    angle_color_right = (0, 165, 255)  # Orange
                elif right_stage == "up":
                    right_form_feedback = f"RIGHT: Lower arm ({int(right_angle)})"
                    angle_color_right = (0, 255, 0)
                else:
                    right_form_feedback = f"RIGHT: Go to 90 ({int(right_angle)})"
                    angle_color_right = (128, 128, 128)
        
        # ========== DRAW POINTS AND CONNECTIONS ==========
        for idx, (name, color) in arm_points.items():
            if idx in point_coords:
                cx, cy = point_coords[idx]
                cv2.circle(frame, (cx, cy), 10, color, -1)
                cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
        
        # Draw connections between joints
        connections = [
            (11, 12, (255, 255, 255)),    # Shoulder to Shoulder
            (11, 13, angle_color_left),   # Left shoulder to elbow
            (12, 14, angle_color_right),  # Right shoulder to elbow
            (13, 15, angle_color_left),   # Left elbow to wrist
            (14, 16, angle_color_right),  # Right elbow to wrist
        ]
        
        for start_idx, end_idx, color in connections:
            if start_idx in point_coords and end_idx in point_coords:
                cv2.line(frame, point_coords[start_idx], point_coords[end_idx], color, 4)
        
        # ========== DISPLAY ANGLE AT ELBOW ==========
        if left_angle and 13 in point_coords:
            cx, cy = point_coords[13]
            cv2.putText(frame, f"{int(left_angle)}", (cx - 30, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, angle_color_left, 2)
        
        if right_angle and 14 in point_coords:
            cx, cy = point_coords[14]
            cv2.putText(frame, f"{int(right_angle)}", (cx - 30, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, angle_color_right, 2)

    # ========== DISPLAY INFO ON SCREEN ==========
    # Create overlay for info
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Title
    cv2.putText(frame, "GYM TRAINER - SHOULDER PRESS", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Rep count
    cv2.putText(frame, f"LEFT ARM:  {left_rep_count} reps", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"RIGHT ARM: {right_rep_count} reps", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Total reps
    cv2.putText(frame, f"TOTAL: {left_rep_count + right_rep_count} reps", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Form feedback
    if left_form_feedback:
        color = (0, 255, 0) if "COMPLETE" in left_form_feedback or "READY" in left_form_feedback else (0, 165, 255)
        cv2.putText(frame, left_form_feedback, (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if right_form_feedback:
        color = (0, 255, 0) if "COMPLETE" in right_form_feedback or "READY" in right_form_feedback else (0, 165, 255)
        cv2.putText(frame, right_form_feedback, (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Key instructions
    cv2.putText(frame, "R: Reset | Q: Quit", (w - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Angle guide
    cv2.putText(frame, "85-95 -> 145+ = 1 REP", (w - 200, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Display result
    cv2.imshow("Gym Trainer - Shoulder Exercise", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        left_rep_count = 0
        right_rep_count = 0
        left_stage = "down"
        right_stage = "down"
        print("\n[RESET] Rep count reset to 0\n")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("FINAL RESULTS:")
print(f"  Left arm:  {left_rep_count} reps")
print(f"  Right arm: {right_rep_count} reps")
print(f"  Total:     {left_rep_count + right_rep_count} reps")
print("=" * 50)
