import cv2
import mediapipe as mp
import os
import random
import math
import time
import subprocess
import sys

def select_folder_mac():
    """Opens a native macOS folder picker using AppleScript, ensuring it comes to front."""
    try:
        # Script to force focus and ask for folder
        script = '''
        tell application "System Events"
            activate
            set f to choose folder with prompt "Select Folder to Snap"
            return POSIX path of f
        end tell
        '''
        cmd = ['osascript', '-e', script]
        print("Waiting for user to select folder...")
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return result.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        print("Folder selection cancelled or failed.")
        return None

def perform_snap_action(selected_folder):
    try:
        files = [f for f in os.listdir(selected_folder) if os.path.isfile(os.path.join(selected_folder, f))]
        
        if not files:
            return 0

        # Shuffle and pick 50%
        random.shuffle(files)
        num_to_delete = math.ceil(len(files) / 2)
        files_to_delete = files[:num_to_delete]

        print(f"Snapping {len(files_to_delete)} files...")
        
        for f in files_to_delete:
            file_path = os.path.join(selected_folder, f)
            try:
                os.remove(file_path)
                print(f"Deleted: {f}")
            except Exception as e:
                print(f"Error deleting {f}: {e}")
        
        return len(files_to_delete)
    except Exception as e:
        print(f"Error accessing folder: {e}")
        return 0

def draw_button(img, text, position, size=(200, 50), color=(0, 255, 0)):
    x, y = position
    w, h = size
    # Scale font and thickness for smaller window
    cv2.rectangle(img, (x, y), (x + w, y + h), color, cv2.FILLED)
    cv2.putText(img, text, (x + 10, y + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
    return (x, y, x + w, y + h)

def is_point_in_rect(point, rect):
    px, py = point
    rx1, ry1, rx2, ry2 = rect
    return rx1 <= px <= rx2 and ry1 <= py <= ry2

def main():
    # State
    folder_path = None
    app_state = "MENU" # MENU, RUNNING, SNAPPED
    
    # Setup Camera
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    # Mouse Callback
    mouse_pos = (0, 0)
    clicked = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_pos, clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pos = (x, y)
            clicked = True

    cv2.namedWindow("Thanos Snap")
    cv2.setMouseCallback("Thanos Snap", mouse_callback)
    
    deleted_count = 0
    snap_start_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Mirror
        img = cv2.flip(img, 1)
        
        # RESIZE 50%
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        h, w, c = img.shape
        # New dimensions approx 320x240 if original was 640x480

        if app_state == "MENU":
            # Draw UI (Adjusted for smaller size)
            cv2.putText(img, "THANOS SNAP", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            
            # Select Folder Button
            # Pos (30, 60), Size (150, 40)
            btn_folder_rect = draw_button(img, "Folder", (30, 60), (150, 40), (200, 200, 0))
            
            # Show Selected Folder
            if folder_path:
                short_path = os.path.basename(folder_path)
                if len(short_path) > 15: short_path = short_path[:12] + "..."
                cv2.putText(img, f"Sel: {short_path}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Start Button
                # Pos (30, 140), Size (150, 40)
                btn_start_rect = draw_button(img, "START", (30, 140), (150, 40), (0, 0, 255))
            else:
                cv2.putText(img, "No Folder", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            # Interactions
            if clicked:
                clicked = False
                if is_point_in_rect(mouse_pos, btn_folder_rect):
                    # Pause CV to show dialog
                    print("Opening folder picker...")
                    # Hack: Release mouse callback temporarily or just blocking call
                    # We need to render the frame once so the user sees the button press?
                    # Python blocks here, so window might freeze. 
                    # cv2.waitKey(1)
                    folder = select_folder_mac()
                    if folder:
                        folder_path = folder
                        print(f"Selected: {folder_path}")
                
                if folder_path and 'btn_start_rect' in locals() and is_point_in_rect(mouse_pos, btn_start_rect):
                    app_state = "RUNNING"
                    clicked = False

        elif app_state == "RUNNING":
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            cv2.putText(img, "PINCH TO SNAP!", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    lmList = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        height, width, _ = img.shape
                        cx, cy = int(lm.x * width), int(lm.y * height)
                        lmList.append([id, cx, cy])

                    if len(lmList) != 0:
                        x1, y1 = lmList[4][1], lmList[4][2] # Thumb
                        x2, y2 = lmList[8][1], lmList[8][2] # Index (Changed from Middle)
                        
                        length = math.hypot(x2 - x1, y2 - y1)
                        
                        cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
                        cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                        # Logic: Distance < 15 (Touching)
                        if length < 15: 
                            app_state = "SNAPPED"
                            snap_start_time = time.time()
                            deleted_count = perform_snap_action(folder_path)

        elif app_state == "SNAPPED":
            cv2.putText(img, "SNAP!", (50, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
            cv2.putText(img, f"-{deleted_count} files", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if time.time() - snap_start_time > 4.0:
                 folder_path = None
                 app_state = "MENU"

        cv2.imshow("Thanos Snap", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
