"""
Fell Sleep Detection - Phát hiện ngủ gật
Sử dụng MediaPipe để nhận diện khuôn mặt và phát hiện khi mắt nhắm quá 3 giây
"""

import cv2
import mediapipe as mp
import time
import numpy as np

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Các điểm mốc cho mắt trái và mắt phải
# Mắt trái: điểm trên và dưới
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

# Mắt phải: điểm trên và dưới  
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Các điểm viền mắt để vẽ
LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Ngưỡng EAR (Eye Aspect Ratio) để xác định mắt nhắm
EAR_THRESHOLD = 0.2
# Thời gian mắt nhắm để cảnh báo (giây)
SLEEP_TIME_THRESHOLD = 3.0


def calculate_ear(landmarks, eye_indices, frame_width, frame_height):
    """
    Tính Eye Aspect Ratio (EAR) cho một mắt
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    # Lấy tọa độ các điểm mắt
    def get_point(idx):
        return np.array([
            landmarks[idx].x * frame_width,
            landmarks[idx].y * frame_height
        ])
    
    if eye_indices == "left":
        p1 = get_point(LEFT_EYE_LEFT)
        p2 = get_point(159)  # top-left
        p3 = get_point(158)  # top-right
        p4 = get_point(LEFT_EYE_RIGHT)
        p5 = get_point(153)  # bottom-right
        p6 = get_point(145)  # bottom-left
    else:  # right
        p1 = get_point(RIGHT_EYE_LEFT)
        p2 = get_point(386)  # top-left
        p3 = get_point(387)  # top-right
        p4 = get_point(RIGHT_EYE_RIGHT)
        p5 = get_point(380)  # bottom-right
        p6 = get_point(374)  # bottom-left
    
    # Tính khoảng cách Euclidean
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    
    # Tính EAR
    if horizontal == 0:
        return 0
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def draw_eye_landmarks(frame, landmarks, eye_indices, frame_width, frame_height, color):
    """Vẽ các điểm mốc của mắt"""
    points = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * frame_width)
        y = int(landmarks[idx].y * frame_height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 2, color, -1)
    
    # Vẽ đường nối các điểm
    if len(points) > 1:
        for i in range(len(points)):
            cv2.line(frame, points[i], points[(i + 1) % len(points)], color, 1)


def main():
    # Khởi tạo webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Biến theo dõi trạng thái mắt nhắm
    eyes_closed_start = None
    is_sleeping = False
    
    # Biến đếm chớp mắt
    left_blink_count = 0
    right_blink_count = 0
    left_eye_was_closed = False
    right_eye_was_closed = False
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Không thể đọc từ webcam")
                continue
            
            # Lật frame để tạo hiệu ứng gương
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Chuyển đổi màu BGR sang RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Xử lý với MediaPipe
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    # Vẽ tất cả các điểm mốc trên khuôn mặt
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Vẽ viền mặt
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    # Vẽ mống mắt
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )
                    
                    # Vẽ các điểm mốc mắt với màu đặc biệt
                    draw_eye_landmarks(frame, landmarks, LEFT_EYE_LANDMARKS, 
                                       frame_width, frame_height, (0, 255, 0))
                    draw_eye_landmarks(frame, landmarks, RIGHT_EYE_LANDMARKS, 
                                       frame_width, frame_height, (0, 255, 0))
                    
                    # Tính EAR cho cả hai mắt
                    # Lưu ý: Do flip frame nên left trong landmarks = right trên màn hình
                    left_ear_raw = calculate_ear(landmarks, "left", frame_width, frame_height)
                    right_ear_raw = calculate_ear(landmarks, "right", frame_width, frame_height)
                    
                    # Swap để khớp với góc nhìn người dùng (do flip)
                    left_ear = right_ear_raw   # Mắt trái người dùng = right landmarks
                    right_ear = left_ear_raw   # Mắt phải người dùng = left landmarks
                    avg_ear = (left_ear + right_ear) / 2
                    
                    # Đếm chớp mắt trái (từ góc nhìn người dùng)
                    if left_ear < EAR_THRESHOLD:
                        left_eye_was_closed = True
                    else:
                        if left_eye_was_closed:
                            left_blink_count += 1
                        left_eye_was_closed = False
                    
                    # Đếm chớp mắt phải (từ góc nhìn người dùng)
                    if right_ear < EAR_THRESHOLD:
                        right_eye_was_closed = True
                    else:
                        if right_eye_was_closed:
                            right_blink_count += 1
                        right_eye_was_closed = False
                    
                    # Hiển thị EAR
                    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Hiển thị EAR từng mắt (từ góc nhìn người dùng)
                    cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
                    cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (30, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
                    
                    # Hiển thị số lần chớp mắt
                    cv2.putText(frame, f"Left Blinks: {left_blink_count}", (30, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
                    cv2.putText(frame, f"Right Blinks: {right_blink_count}", (30, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
                    cv2.putText(frame, f"Total Blinks: {left_blink_count + right_blink_count}", (30, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Kiểm tra mắt có nhắm không
                    if avg_ear < EAR_THRESHOLD:
                        if eyes_closed_start is None:
                            eyes_closed_start = time.time()
                        else:
                            closed_duration = time.time() - eyes_closed_start
                            cv2.putText(frame, f"Mat nham: {closed_duration:.1f}s", (30, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            
                            # Kiểm tra nếu nhắm mắt quá 3 giây
                            if closed_duration >= SLEEP_TIME_THRESHOLD:
                                is_sleeping = True
                    else:
                        eyes_closed_start = None
                        is_sleeping = False
                    
                    # Hiển thị cảnh báo FELL SLEEP
                    if is_sleeping:
                        # Vẽ overlay đỏ nhấp nháy
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 0, 255), -1)
                        alpha = 0.3 + 0.2 * abs(np.sin(time.time() * 5))
                        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                        
                        # Vẽ text cảnh báo lớn
                        text = "FELL SLEEP!"
                        font = cv2.FONT_HERSHEY_DUPLEX
                        font_scale = 3
                        thickness = 5
                        
                        # Lấy kích thước text
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        # Tính vị trí giữa màn hình
                        x = (frame_width - text_width) // 2
                        y = (frame_height + text_height) // 2
                        
                        # Vẽ viền đen cho text
                        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 3)
                        # Vẽ text trắng
                        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
                        
                        # Vẽ cảnh báo phụ
                        warning_text = "THUC DAY! THUC DAY!"
                        (w_width, w_height), _ = cv2.getTextSize(warning_text, font, 1, 2)
                        cv2.putText(frame, warning_text, 
                                    ((frame_width - w_width) // 2, y + 60),
                                    font, 1, (0, 255, 255), 2)
                    
                    # Hiển thị trạng thái mắt
                    status = "Mat NHAM" if avg_ear < EAR_THRESHOLD else "Mat MO"
                    color = (0, 0, 255) if avg_ear < EAR_THRESHOLD else (0, 255, 0)
                    cv2.putText(frame, status, (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "Khong tim thay khuon mat", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Hiển thị hướng dẫn
            cv2.putText(frame, "Nhan 'Q' de thoat", (frame_width - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Hiển thị frame
            cv2.imshow('Fell Sleep Detection', frame)
            
            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()