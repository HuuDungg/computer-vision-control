import cv2
import os
import time
from datetime import datetime
import numpy as np

# Configuration
FACE_DIR = "face"
LOG_FILE = "presence_log.txt"
TRAINER_FILE = "trainer.yml"

# Initialize Face Recognizer and Cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels():
    """
    Load images from FACE_DIR, detect faces, and return faces list and ids list.
    Also returns a mapping of id -> name.
    """
    faces = []
    ids = []
    names = {}
    current_id = 0

    if not os.path.exists(FACE_DIR):
        os.makedirs(FACE_DIR)
        return [], [], {}

    print("Training faces...")
    
    # Iterate through user directories
    for user_name in os.listdir(FACE_DIR):
        user_dir = os.path.join(FACE_DIR, user_name)
        if not os.path.isdir(user_dir):
            continue
        
        # Assign ID to name
        # We need integer IDs for LBPH
        user_id = current_id
        names[user_id] = user_name
        current_id += 1
        
        has_images = False
        for filename in os.listdir(user_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(user_dir, filename)
                try:
                    # Read image and convert to grayscale
                    img = cv2.imread(image_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Detect face in the training image
                    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    
                    for (x, y, w, h) in faces_rect:
                        faces.append(gray[y:y+h, x:x+w])
                        ids.append(user_id)
                        has_images = True
                except Exception as e:
                    print(f"Skipping {image_path}: {e}")
        
        if not has_images:
            # Revert ID if no valid images found for this user
            current_id -= 1
            del names[user_id]

    print(f"Training complete. {len(faces)} face samples loaded for {len(names)} users.")
    return faces, ids, names

def train_model():
    """
    Train the LBPH recognizer and return the name mapping.
    """
    faces, ids, names = get_images_and_labels()
    if not faces:
        return {}
    
    recognizer.train(faces, np.array(ids))
    return names

def register_new_user(cap):
    """
    Capture images for a new user and save them to FACE_DIR.
    """
    print("\n--- NEW USER REGISTRATION ---")
    name = input("Enter your name: ").strip()
    if not name:
        return {}

    user_dir = os.path.join(FACE_DIR, name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    print("Please look at the camera. Capturing 15 samples...")
    count = 0
    while count < 15:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Save the captured face
            # Save specifically the face region or the whole frame? 
            # LBPH pipeline in get_images_and_labels re-detects, so saving whole frame is fine, 
            # but saving crop is safer if detection varies. 
            # Let's save the whole frame as typical.
            cv2.imwrite(os.path.join(user_dir, f"{name}_{count}.jpg"), frame)
            count += 1
            print(f"Captured {count}/15")
            time.sleep(0.2) 

        cv2.imshow('Registration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Registration complete for {name}. Retraining model...")
    return train_model()

def log_status(user, status, duration):
    """
    Log the status change to a file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] User: {user}, Status: {status}, Duration: {duration:.2f}s\n"
    print(log_entry.strip())
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)

def main():
    cap = cv2.VideoCapture(0)
    wCam, hCam = 640, 480
    cap.set(3, wCam)
    cap.set(4, hCam)

    # Initial Training
    names = train_model()
    
    # If no model trained (no data), force registration
    if not names:
        print("No registered faces found.")
        names = register_new_user(cap)

    current_status = "OFF_SCREEN"
    current_user_name = "Unknown"
    status_start_time = time.time()
    
    # Confidence threshold (lower is better for LBPH, unlike others)
    # Typical: 0 (perfect) to ~100. < 50-60 is usually a match.
    CONFIDENCE_THRESHOLD = 60 

    print("\nStarting Presence Detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1) # Mirror
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        
        detected_name = "Unknown"
        
        for (x, y, w, h) in faces:
            # Predict
            try:
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                # Check confidence
                if confidence < 100:
                    name = names.get(id, "Unknown")
                    confidence_text = f"  {round(100 - confidence)}%"
                else:
                    name = "Unknown"
                    confidence_text = f"  {round(100 - confidence)}%"
                
                if confidence < CONFIDENCE_THRESHOLD:
                    detected_name = name
                    color = (0, 255, 0)
                else:
                    detected_name = "Unknown"
                    color = (0, 0, 255)
            
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, str(name), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
            except Exception as e:
                pass # Model might not be trained if registration was skipped/failed

        # Status Logic
        is_known_present = (detected_name != "Unknown")
        new_status = "ON_SCREEN" if is_known_present else "OFF_SCREEN"
        
        if new_status != current_status:
            end_time = time.time()
            duration = end_time - status_start_time
            
            # Log previous state
            if current_status == "ON_SCREEN":
                log_name = current_user_name
            else:
                log_name = detected_name if detected_name != "Unknown" else "System"

            if duration > 1.0: # Filter brief flickers (optional but good)
                log_status(log_name, current_status, duration)
                current_status = new_status
                status_start_time = end_time
                if is_known_present:
                    current_user_name = detected_name

        # UI
        cv2.putText(frame, f"Status: {current_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        current_dur = time.time() - status_start_time
        cv2.putText(frame, f"Time: {current_dur:.1f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Presence Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
